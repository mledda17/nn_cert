import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


class VariableStore:
    def __init__(self):
        self.lower_bounds = []
        self.upper_bounds = []
        self.integrality = []
        self.names = {}

    def add(self, name, size, lower_bound, upper_bound, integer=False):
        start = len(self.lower_bounds)
        indexes = np.arange(start, start + size)
        self.names[name] = indexes

        if np.isscalar(lower_bound):
            lower_values = np.full(size, lower_bound, dtype=float)
        else:
            lower_values = np.array(lower_bound, dtype=float).reshape(-1)

        if np.isscalar(upper_bound):
            upper_values = np.full(size, upper_bound, dtype=float)
        else:
            upper_values = np.array(upper_bound, dtype=float).reshape(-1)

        self.lower_bounds.extend(lower_values.tolist())
        self.upper_bounds.extend(upper_values.tolist())

        value = 1 if integer else 0
        self.integrality.extend([value] * size)
        return indexes

    def size(self):
        return len(self.lower_bounds)


class ConstraintStore:
    def __init__(self):
        self.rows = []
        self.cols = []
        self.data = []
        self.lower_bounds = []
        self.upper_bounds = []

    def add(self, terms, lower_bound, upper_bound):
        row = len(self.lower_bounds)
        for variable_index, coefficient in terms:
            if coefficient != 0.0:
                self.rows.append(row)
                self.cols.append(int(variable_index))
                self.data.append(float(coefficient))
        self.lower_bounds.append(float(lower_bound))
        self.upper_bounds.append(float(upper_bound))

    def to_linear_constraint(self, num_variables):
        matrix = coo_matrix(
            (self.data, (self.rows, self.cols)),
            shape=(len(self.lower_bounds), num_variables),
        ).tocsr()

        return LinearConstraint(
            matrix,
            np.array(self.lower_bounds, dtype=float),
            np.array(self.upper_bounds, dtype=float),
        )


def compute_network_bounds(weights, biases, input_lower, input_upper):
    z_lowers = []
    z_uppers = []
    h_lowers = []
    h_uppers = []

    lower = np.array(input_lower, dtype=float).reshape(-1)
    upper = np.array(input_upper, dtype=float).reshape(-1)

    for layer_index in range(len(weights)):
        weight = np.array(weights[layer_index], dtype=float)
        bias = np.array(biases[layer_index], dtype=float)

        positive_weight = np.maximum(weight, 0.0)
        negative_weight = np.minimum(weight, 0.0)

        z_lower = positive_weight @ lower + negative_weight @ upper + bias
        z_upper = positive_weight @ upper + negative_weight @ lower + bias

        z_lowers.append(z_lower)
        z_uppers.append(z_upper)

        is_output_layer = layer_index == len(weights) - 1
        if is_output_layer:
            lower = z_lower
            upper = z_upper
        else:
            lower = np.maximum(z_lower, 0.0)
            upper = np.maximum(z_upper, 0.0)

        h_lowers.append(lower)
        h_uppers.append(upper)

    return z_lowers, z_uppers, h_lowers, h_uppers


def estimate_error_bound(reference_bounds, candidate_bounds):
    ref_lower = reference_bounds[2][-1]
    ref_upper = reference_bounds[3][-1]
    cand_lower = candidate_bounds[2][-1]
    cand_upper = candidate_bounds[3][-1]

    error_lower = cand_lower - ref_upper
    error_upper = cand_upper - ref_lower
    absolute_error = np.maximum(np.abs(error_lower), np.abs(error_upper))

    return float(np.max(absolute_error) + 1e-6)


def add_network_constraints(
    variables,
    constraints,
    prefix,
    input_variables,
    output_variables,
    weights,
    biases,
    network_bounds,
):
    z_lowers, z_uppers, h_lowers, h_uppers = network_bounds
    previous_variables = input_variables

    for layer_index in range(len(weights)):
        weight = np.array(weights[layer_index], dtype=float)
        bias = np.array(biases[layer_index], dtype=float)
        layer_size = bias.shape[0]

        z_variables = variables.add(
            prefix + "_z_" + str(layer_index),
            layer_size,
            z_lowers[layer_index],
            z_uppers[layer_index],
        )

        for i in range(layer_size):
            terms = [(z_variables[i], 1.0)]
            for j in range(weight.shape[1]):
                terms.append((previous_variables[j], -weight[i, j]))
            constraints.add(terms, bias[i], bias[i])

        is_output_layer = layer_index == len(weights) - 1
        if is_output_layer:
            for i in range(layer_size):
                constraints.add(
                    [(output_variables[i], 1.0), (z_variables[i], -1.0)],
                    0.0,
                    0.0,
                )
            previous_variables = output_variables
        else:
            h_variables = variables.add(
                prefix + "_h_" + str(layer_index),
                layer_size,
                h_lowers[layer_index],
                h_uppers[layer_index],
            )

            for i in range(layer_size):
                z_lower = z_lowers[layer_index][i]
                z_upper = z_uppers[layer_index][i]

                if z_upper <= 0.0:
                    constraints.add([(h_variables[i], 1.0)], 0.0, 0.0)
                elif z_lower >= 0.0:
                    constraints.add(
                        [(h_variables[i], 1.0), (z_variables[i], -1.0)],
                        0.0,
                        0.0,
                    )
                else:
                    delta = variables.add(
                        prefix + "_delta_" + str(layer_index) + "_" + str(i),
                        1,
                        0.0,
                        1.0,
                        integer=True,
                    )

                    constraints.add(
                        [(h_variables[i], 1.0), (z_variables[i], -1.0)],
                        0.0,
                        np.inf,
                    )
                    constraints.add(
                        [(h_variables[i], 1.0), (delta[0], -z_upper)],
                        -np.inf,
                        0.0,
                    )
                    constraints.add(
                        [
                            (h_variables[i], 1.0),
                            (z_variables[i], -1.0),
                            (delta[0], -z_lower),
                        ],
                        -np.inf,
                        -z_lower,
                    )

            previous_variables = h_variables


def build_information_bounds(window_length, y_min, y_max, u_min, u_max, ny, nu):
    lower = []
    upper = []

    for _ in range(window_length):
        for _ in range(ny):
            lower.append(y_min)
            upper.append(y_max)
        for _ in range(nu):
            lower.append(u_min)
            upper.append(u_max)

    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def add_information_dynamics(
    constraints,
    information_variables,
    reference_output_variables,
    input_variables,
    window_length,
    ny,
    nu,
):
    q = ny + nu
    information_size = window_length * q
    last_pair_start = (window_length - 1) * q

    for k in range(len(information_variables) - 1):
        current_information = information_variables[k]
        next_information = information_variables[k + 1]

        for i in range(information_size - q):
            constraints.add(
                [(next_information[i], 1.0), (current_information[i + q], -1.0)],
                0.0,
                0.0,
            )

        for i in range(ny):
            constraints.add(
                [
                    (next_information[last_pair_start + i], 1.0),
                    (reference_output_variables[k][i], -1.0),
                ],
                0.0,
                0.0,
            )

        for i in range(nu):
            constraints.add(
                [
                    (next_information[last_pair_start + ny + i], 1.0),
                    (input_variables[k][i], -1.0),
                ],
                0.0,
                0.0,
            )


def solve_certification_milp(
    reference_weights,
    reference_biases,
    candidate_weights,
    candidate_biases,
    horizon,
    window_length=10,
    y_min=0.0,
    y_max=10.0,
    u_min=-2.0,
    u_max=2.0,
    information_lower=None,
    information_upper=None,
    time_limit=600.0,
    num_threads=0,
    show_solver_output=False,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    ny = reference_biases[-1].shape[0]
    reference_input_size = reference_weights[0].shape[1]
    candidate_input_size = candidate_weights[0].shape[1]

    if reference_input_size != candidate_input_size:
        raise ValueError("reference and candidate networks must have the same input size")
    if reference_input_size % window_length != 0:
        raise ValueError("network input size must be divisible by window_length")

    signals_per_window_step = reference_input_size // window_length
    nu = signals_per_window_step - ny
    if nu <= 0:
        raise ValueError("could not infer a positive input dimension")

    q = ny + nu
    information_size = window_length * q
    last_time = window_length + horizon - 1

    if information_lower is None or information_upper is None:
        information_lower, information_upper = build_information_bounds(
            window_length,
            y_min,
            y_max,
            u_min,
            u_max,
            ny,
            nu,
        )
    else:
        information_lower = np.array(information_lower, dtype=float).reshape(-1)
        information_upper = np.array(information_upper, dtype=float).reshape(-1)
        if information_lower.shape[0] != information_size:
            raise ValueError("information_lower has the wrong size")
        if information_upper.shape[0] != information_size:
            raise ValueError("information_upper has the wrong size")
        if np.any(information_lower > information_upper):
            raise ValueError("information_lower must be less than or equal to information_upper")

    reference_bounds = compute_network_bounds(
        reference_weights,
        reference_biases,
        information_lower,
        information_upper,
    )
    candidate_bounds = compute_network_bounds(
        candidate_weights,
        candidate_biases,
        information_lower,
        information_upper,
    )
    error_bound = estimate_error_bound(reference_bounds, candidate_bounds)

    variables = VariableStore()
    constraints = ConstraintStore()

    information_variables = []
    for k in range(last_time + 1):
        information_variables.append(
            variables.add(
                "I_" + str(k),
                information_size,
                information_lower,
                information_upper,
            )
        )

    input_variables = []
    for k in range(last_time):
        input_variables.append(variables.add("u_" + str(k), nu, u_min, u_max))

    reference_output_variables = []
    for k in range(last_time + 1):
        reference_output_variables.append(variables.add("y_" + str(k), ny, y_min, y_max))

    candidate_output_variables = {}
    epsilon_variables = {}
    beta_plus_variables = {}
    beta_minus_variables = {}

    for k in range(window_length, last_time + 1):
        candidate_output_variables[k] = variables.add(
            "y_hat_" + str(k),
            ny,
            -np.inf,
            np.inf,
        )
        epsilon_variables[k] = variables.add("epsilon_" + str(k), 1, 0.0, error_bound)
        beta_plus_variables[k] = variables.add(
            "beta_plus_" + str(k),
            ny,
            0.0,
            1.0,
            integer=True,
        )
        beta_minus_variables[k] = variables.add(
            "beta_minus_" + str(k),
            ny,
            0.0,
            1.0,
            integer=True,
        )

    add_information_dynamics(
        constraints,
        information_variables,
        reference_output_variables,
        input_variables,
        window_length,
        ny,
        nu,
    )

    for k in range(last_time + 1):
        add_network_constraints(
            variables,
            constraints,
            "ref_" + str(k),
            information_variables[k],
            reference_output_variables[k],
            reference_weights,
            reference_biases,
            reference_bounds,
        )

    for k in range(window_length, last_time + 1):
        add_network_constraints(
            variables,
            constraints,
            "cand_" + str(k),
            information_variables[k],
            candidate_output_variables[k],
            candidate_weights,
            candidate_biases,
            candidate_bounds,
        )

    for k in range(window_length, last_time + 1):
        for i in range(ny):
            epsilon = epsilon_variables[k][0]
            reference_output = reference_output_variables[k][i]
            candidate_output = candidate_output_variables[k][i]
            beta_plus = beta_plus_variables[k][i]
            beta_minus = beta_minus_variables[k][i]

            constraints.add(
                [(epsilon, 1.0), (candidate_output, -1.0), (reference_output, 1.0)],
                0.0,
                np.inf,
            )
            constraints.add(
                [(epsilon, 1.0), (candidate_output, 1.0), (reference_output, -1.0)],
                0.0,
                np.inf,
            )
            constraints.add(
                [
                    (epsilon, 1.0),
                    (candidate_output, -1.0),
                    (reference_output, 1.0),
                    (beta_plus, 2.0 * error_bound),
                ],
                -np.inf,
                2.0 * error_bound,
            )
            constraints.add(
                [
                    (epsilon, 1.0),
                    (candidate_output, 1.0),
                    (reference_output, -1.0),
                    (beta_minus, 2.0 * error_bound),
                ],
                -np.inf,
                2.0 * error_bound,
            )

        beta_terms = []
        for i in range(ny):
            beta_terms.append((beta_plus_variables[k][i], 1.0))
            beta_terms.append((beta_minus_variables[k][i], 1.0))
        constraints.add(beta_terms, 1.0, 1.0)

    objective = np.zeros(variables.size(), dtype=float)
    for k in range(window_length, last_time + 1):
        objective[epsilon_variables[k][0]] = -1.0

    if show_solver_output:
        print("MILP variables:", variables.size(), flush=True)
        print("MILP constraints:", len(constraints.lower_bounds), flush=True)
        print("MILP binary variables:", int(sum(variables.integrality)), flush=True)
        print("MILP error bound P:", error_bound, flush=True)

    solver_options = {"time_limit": time_limit, "disp": show_solver_output}
    if num_threads > 0:
        solver_options["threads"] = num_threads

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected: .*",
            category=RuntimeWarning,
        )
        result = milp(
            c=objective,
            integrality=np.array(variables.integrality, dtype=int),
            bounds=Bounds(
                np.array(variables.lower_bounds, dtype=float),
                np.array(variables.upper_bounds, dtype=float),
            ),
            constraints=constraints.to_linear_constraint(variables.size()),
            options=solver_options,
        )

    if result.fun is None:
        objective_value = None
    else:
        objective_value = -float(result.fun)

    return {
        "result": result,
        "objective_value": objective_value,
        "error_bound": error_bound,
        "variables": variables,
        "constraints": constraints,
        "information_variables": information_variables,
        "reference_output_variables": reference_output_variables,
        "candidate_output_variables": candidate_output_variables,
        "epsilon_variables": epsilon_variables,
    }

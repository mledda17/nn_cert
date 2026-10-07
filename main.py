import copy
import os
import random

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Numerical setup from the paper.
GUROBI_LICENSE_FILE = "/Users/marco/Downloads/gurobi (3).lic"
if "GRB_LICENSE_FILE" not in os.environ and os.path.exists(GUROBI_LICENSE_FILE):
    os.environ["GRB_LICENSE_FILE"] = GUROBI_LICENSE_FILE

from milp_certification import solve_certification_milp

T = 5
CERTIFICATION_HORIZON = 5
HORIZON_SWEEP = True
SOLVER_BACKEND = "gurobi"
REFERENCE_HIDDEN_LAYERS = 1
REFERENCE_NEURONS = 16
CANDIDATE_HIDDEN_LAYERS = 1
CANDIDATE_NEURONS = 8
MILP_TIME_LIMIT = 600.0
MILP_THREADS = 8
MILP_RELATIVE_GAP = 1e-2
MILP_BIG_M = None
MILP_P = None
DATA_BOUNDS_MARGIN = 0.05
GRADIENT_ASCENT_RESTARTS = 20
GRADIENT_ASCENT_STEPS = 300
GRADIENT_ASCENT_LR = 0.05
N_TRAIN = 20_000
N_VAL = 2_000
NOISE_STD = 0.02

U_MIN = -2.0
U_MAX = 2.0
X_MIN = 0.0
X_MAX = 5.0

K1 = 0.5
K2 = 0.4
K3 = 0.2
K4 = 0.3

SEED = 1
MODEL_FILE = "trained_models.pt"
FORCE_RETRAIN = False
TRAJECTORY_PLOT_FILE = "output_trajectory.png"
INPUT_PLOT_FILE = "input_sequence.png"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cascaded_tanks_step(x, u):
    x1 = max(float(x[0]), 0.0)
    x2 = max(float(x[1]), 0.0)

    next_x1 = x1 - K1 * np.sqrt(x1) + K2 * float(u)
    next_x2 = x2 + K3 * np.sqrt(x1) - K4 * np.sqrt(x2)

    next_x1 = max(next_x1, X_MIN)
    next_x2 = max(next_x2, X_MIN)
    return np.array([next_x1, next_x2], dtype=np.float32)


def cascaded_tanks_output(x):
    return np.array([x[1]], dtype=np.float32)


def sample_piecewise_constant_inputs(num_steps, segment_length):
    values = []
    while len(values) < num_steps:
        amplitude = np.random.uniform(U_MIN, U_MAX)
        for _ in range(segment_length):
            values.append(amplitude)
    return np.array(values[:num_steps], dtype=np.float32)


def simulate_system(num_steps):
    x = np.random.uniform(X_MIN, X_MAX, size=2).astype(np.float32)
    u_values = sample_piecewise_constant_inputs(num_steps, T)

    y_values = []
    for k in range(num_steps):
        y_values.append(cascaded_tanks_output(x))
        x = cascaded_tanks_step(x, u_values[k])

    y_values = np.array(y_values, dtype=np.float32)
    u_values = u_values.reshape(-1, 1)
    return y_values, u_values


def make_sample(y_values, u_values, start_index):
    information = []
    for k in range(start_index, start_index + T):
        information.append(y_values[k, 0])
        information.append(u_values[k, 0])
    target = y_values[start_index + T, 0]
    return information, target


def generate_dataset(num_samples):
    num_steps = num_samples + T
    y_clean, u_clean = simulate_system(num_steps)

    y_noisy = y_clean + np.random.normal(0.0, NOISE_STD, size=y_clean.shape)
    u_noisy = u_clean + np.random.normal(0.0, NOISE_STD, size=u_clean.shape)

    x_data = []
    y_data = []
    for i in range(num_samples):
        information, target = make_sample(y_noisy, u_noisy, i)
        x_data.append(information)
        y_data.append(target)

    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32).reshape(-1, 1)
    return x_data, y_data


def compute_normalization(train_x, train_y):
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True)
    y_mean = train_y.mean(axis=0, keepdims=True)
    y_std = train_y.std(axis=0, keepdims=True)

    x_std[x_std < 1e-8] = 1.0
    y_std[y_std < 1e-8] = 1.0
    return x_mean, x_std, y_mean, y_std


def normalize_data(x_data, y_data, x_mean, x_std, y_mean, y_std):
    x_norm = (x_data - x_mean) / x_std
    y_norm = (y_data - y_mean) / y_std
    return x_norm.astype(np.float32), y_norm.astype(np.float32)


def compute_information_bounds_from_data(x_data, margin):
    lower = x_data.min(axis=0)
    upper = x_data.max(axis=0)
    width = upper - lower

    lower = lower - margin * width
    upper = upper + margin * width

    for i in range(T):
        y_index = 2 * i
        u_index = 2 * i + 1

        lower[y_index] = max(lower[y_index], X_MIN)
        upper[y_index] = min(upper[y_index], X_MAX)
        lower[u_index] = max(lower[u_index], U_MIN)
        upper[u_index] = min(upper[u_index], U_MAX)

    return lower.astype(np.float64), upper.astype(np.float64)


class ReluMlp(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons_per_layer, output_size):
        super().__init__()
        layers = []
        last_size = input_size
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_size, neurons_per_layer))
            layers.append(nn.ReLU())
            last_size = neurons_per_layer
        layers.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total


def train_model(model, train_x, train_y, val_x, val_y, epochs):
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for batch_x, batch_y in train_loader:
            prediction = model(batch_x)
            loss = loss_function(prediction, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.shape[0]

        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                train_loss = train_loss_sum / len(train_dataset)
                val_prediction = model(torch.from_numpy(val_x))
                val_loss = loss_function(val_prediction, torch.from_numpy(val_y)).item()
            print(
                "epoch",
                epoch + 1,
                "of",
                epochs,
                "- train mse:",
                round(train_loss, 6),
                "- val mse:",
                round(val_loss, 6),
            )


def fold_normalization(model, x_mean, x_std, y_mean, y_std):
    folded = copy.deepcopy(model)
    linear_layers = []

    for layer in folded.net:
        if isinstance(layer, nn.Linear):
            linear_layers.append(layer)

    first = linear_layers[0]
    last = linear_layers[-1]

    x_mean_tensor = torch.from_numpy(x_mean.reshape(-1)).float()
    x_std_tensor = torch.from_numpy(x_std.reshape(-1)).float()
    y_mean_tensor = torch.from_numpy(y_mean.reshape(-1)).float()
    y_std_tensor = torch.from_numpy(y_std.reshape(-1)).float()

    with torch.no_grad():
        original_first_weight = first.weight.clone()
        first.weight.copy_(original_first_weight / x_std_tensor)
        normalized_offset = original_first_weight * x_mean_tensor / x_std_tensor
        first.bias.copy_(first.bias - normalized_offset.sum(dim=1))

        last.weight.copy_(last.weight * y_std_tensor.reshape(-1, 1))
        last.bias.copy_(last.bias * y_std_tensor + y_mean_tensor)

    return folded


def evaluate_physical_model(model, x_data, y_data):
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x_data)).numpy()
    mse = np.mean((prediction - y_data) ** 2)
    max_abs_error = np.max(np.abs(prediction - y_data))
    return mse, max_abs_error


def extract_weights_and_biases(model):
    weights = []
    biases = []

    for layer in model.net:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().numpy().copy())
            biases.append(layer.bias.detach().numpy().copy())

    return weights, biases


def checkpoint_matches_setup(checkpoint, input_size, output_size):
    setup = checkpoint.get("setup", {})

    if setup.get("T") != T:
        return False
    if setup.get("input_size") != input_size:
        return False
    if setup.get("output_size") != output_size:
        return False
    if setup.get("reference_hidden_layers") != REFERENCE_HIDDEN_LAYERS:
        return False
    if setup.get("reference_neurons") != REFERENCE_NEURONS:
        return False
    if setup.get("candidate_hidden_layers") != CANDIDATE_HIDDEN_LAYERS:
        return False
    if setup.get("candidate_neurons") != CANDIDATE_NEURONS:
        return False
    if setup.get("x_max") != X_MAX:
        return False
    if setup.get("u_min") != U_MIN:
        return False
    if setup.get("u_max") != U_MAX:
        return False

    return True


def save_models(reference_model, candidate_model, input_size, output_size):
    reference_weights, reference_biases = extract_weights_and_biases(reference_model)
    candidate_weights, candidate_biases = extract_weights_and_biases(candidate_model)

    torch.save(
        {
            "reference": reference_model.state_dict(),
            "candidate": candidate_model.state_dict(),
            "reference_weights": reference_weights,
            "reference_biases": reference_biases,
            "candidate_weights": candidate_weights,
            "candidate_biases": candidate_biases,
            "setup": {
                "T": T,
                "u_min": U_MIN,
                "u_max": U_MAX,
                "x_min": X_MIN,
                "x_max": X_MAX,
                "noise_std": NOISE_STD,
                "input_size": input_size,
                "output_size": output_size,
                "reference_hidden_layers": REFERENCE_HIDDEN_LAYERS,
                "reference_neurons": REFERENCE_NEURONS,
                "candidate_hidden_layers": CANDIDATE_HIDDEN_LAYERS,
                "candidate_neurons": CANDIDATE_NEURONS,
            },
        },
        MODEL_FILE,
    )


def find_initial_information_with_gradient_ascent(
    reference_model,
    candidate_model,
    information_lower,
    information_upper,
    restarts,
    steps,
    learning_rate,
):
    lower = torch.from_numpy(information_lower.astype(np.float32))
    upper = torch.from_numpy(information_upper.astype(np.float32))

    reference_model.eval()
    candidate_model.eval()

    best_information = None
    best_value = -1.0

    for restart in range(restarts):
        random_point = lower + torch.rand_like(lower) * (upper - lower)
        information = random_point.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([information], lr=learning_rate)

        for _ in range(steps):
            reference_output = reference_model(information.reshape(1, -1))
            candidate_output = candidate_model(information.reshape(1, -1))
            error = torch.max(torch.abs(candidate_output - reference_output))
            loss = -error

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                information.copy_(torch.maximum(torch.minimum(information, upper), lower))

        with torch.no_grad():
            reference_output = reference_model(information.reshape(1, -1))
            candidate_output = candidate_model(information.reshape(1, -1))
            error = torch.max(torch.abs(candidate_output - reference_output)).item()

        if error > best_value:
            best_value = error
            best_information = information.detach().numpy().copy()

        print(
            "gradient ascent restart",
            restart + 1,
            "of",
            restarts,
            "- best error:",
            round(best_value, 6),
        )

    return best_information, best_value


def run_certification(
    reference_weights,
    reference_biases,
    candidate_weights,
    candidate_biases,
    information_lower,
    information_upper,
    initial_information,
    horizon,
):
    print("Solving certification MILP with horizon", horizon)
    certification = solve_certification_milp(
        reference_weights,
        reference_biases,
        candidate_weights,
        candidate_biases,
        horizon=horizon,
        window_length=T,
        y_min=X_MIN,
        y_max=X_MAX,
        u_min=U_MIN,
        u_max=U_MAX,
        information_lower=information_lower,
        information_upper=information_upper,
        initial_information=initial_information,
        warmup_steps=T,
        big_m_value=MILP_BIG_M,
        error_bound=MILP_P,
        time_limit=MILP_TIME_LIMIT,
        num_threads=MILP_THREADS,
        mip_relative_gap=MILP_RELATIVE_GAP,
        solver_backend=SOLVER_BACKEND,
        show_solver_output=True,
    )

    milp_result = certification["result"]
    if isinstance(milp_result, dict):
        success = milp_result["success"]
        status = milp_result["status"]
        message = milp_result["message"]
    else:
        success = milp_result.success
        status = milp_result.status
        message = milp_result.message

    print("MILP success:", success)
    print("MILP status:", status)
    print("MILP message:", message)
    print("Certified objective:", certification["objective_value"])
    print("MILP error bound P:", certification["error_bound"])
    return certification


def get_input_sequence_from_certification(certification):
    solution = certification["solution"]
    input_variables = certification["input_variables"]

    if solution is None:
        return None

    input_sequence = []
    for variables in input_variables:
        input_sequence.append(solution[variables].copy())

    return np.array(input_sequence, dtype=np.float32)


def rollout_reference_candidate(reference_model, candidate_model, initial_information, input_sequence):
    information = initial_information.astype(np.float32).copy()
    reference_outputs = []
    candidate_outputs = []

    reference_model.eval()
    candidate_model.eval()

    for k in range(input_sequence.shape[0] + 1):
        information_tensor = torch.from_numpy(information.reshape(1, -1))

        with torch.no_grad():
            reference_output = reference_model(information_tensor).numpy().reshape(-1)
            candidate_output = candidate_model(information_tensor).numpy().reshape(-1)

        reference_outputs.append(reference_output[0])
        candidate_outputs.append(candidate_output[0])

        if k < input_sequence.shape[0]:
            new_information = []
            for i in range(1, T):
                old_index = 2 * i
                new_information.append(information[old_index])
                new_information.append(information[old_index + 1])

            new_information.append(reference_output[0])
            new_information.append(input_sequence[k, 0])
            information = np.array(new_information, dtype=np.float32)

    return np.array(reference_outputs), np.array(candidate_outputs)


def plot_output_trajectory(reference_outputs, candidate_outputs, comparison_start, output_file):
    time_indexes = np.arange(reference_outputs.shape[0])
    candidate_to_plot = candidate_outputs.copy()
    candidate_to_plot[:comparison_start] = np.nan
    error_to_plot = np.abs(candidate_outputs - reference_outputs)
    error_to_plot[:comparison_start] = np.nan

    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(time_indexes, reference_outputs, marker="o", label="reference")
    axes[0].plot(time_indexes, candidate_to_plot, marker="s", label="candidate")
    axes[0].axvline(comparison_start, color="black", linestyle="--", linewidth=1.2, label="certification starts")
    axes[0].set_ylabel("y")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time_indexes, error_to_plot, marker="d", color="tab:red", label="absolute error")
    axes[1].axvline(comparison_start, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("|error|")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def print_input_sequence(input_sequence):
    print("Optimized input sequence:")
    for k in range(input_sequence.shape[0]):
        print("u_" + str(k) + " =", float(input_sequence[k, 0]))


def plot_input_sequence(input_sequence, comparison_start, output_file):
    time_indexes = np.arange(input_sequence.shape[0])

    plt.figure(figsize=(8, 3.2))
    plt.step(time_indexes, input_sequence[:, 0], where="post", label="input")
    plt.plot(time_indexes, input_sequence[:, 0], marker="o", linestyle="None")
    plt.axvline(comparison_start, color="black", linestyle="--", linewidth=1.2, label="certification starts")
    plt.xlabel("k")
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def main():
    set_seed(SEED)

    print("Generating dataset")
    train_x, train_y = generate_dataset(N_TRAIN)
    val_x, val_y = generate_dataset(N_VAL)
    information_lower, information_upper = compute_information_bounds_from_data(
        train_x,
        DATA_BOUNDS_MARGIN,
    )

    x_mean, x_std, y_mean, y_std = compute_normalization(train_x, train_y)
    train_x_norm, train_y_norm = normalize_data(train_x, train_y, x_mean, x_std, y_mean, y_std)
    val_x_norm, val_y_norm = normalize_data(val_x, val_y, x_mean, x_std, y_mean, y_std)

    input_size = train_x.shape[1]
    output_size = train_y.shape[1]

    reference = ReluMlp(
        input_size,
        hidden_layers=REFERENCE_HIDDEN_LAYERS,
        neurons_per_layer=REFERENCE_NEURONS,
        output_size=output_size,
    )
    candidate = ReluMlp(
        input_size,
        hidden_layers=CANDIDATE_HIDDEN_LAYERS,
        neurons_per_layer=CANDIDATE_NEURONS,
        output_size=output_size,
    )

    print("Reference parameters:", count_parameters(reference))
    print("Candidate parameters:", count_parameters(candidate))

    loaded_models = False
    if os.path.exists(MODEL_FILE) and not FORCE_RETRAIN:
        checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
        if checkpoint_matches_setup(checkpoint, input_size, output_size):
            print("Loading trained models from", MODEL_FILE)
            reference_physical = ReluMlp(
                input_size,
                hidden_layers=REFERENCE_HIDDEN_LAYERS,
                neurons_per_layer=REFERENCE_NEURONS,
                output_size=output_size,
            )
            candidate_physical = ReluMlp(
                input_size,
                hidden_layers=CANDIDATE_HIDDEN_LAYERS,
                neurons_per_layer=CANDIDATE_NEURONS,
                output_size=output_size,
            )
            try:
                reference_physical.load_state_dict(checkpoint["reference"])
                candidate_physical.load_state_dict(checkpoint["candidate"])
                loaded_models = True
            except RuntimeError:
                print("Existing model file has incompatible weights")
        else:
            print("Existing model file is not compatible with the current setup")

    if not loaded_models:
        print("Training reference network")
        train_model(reference, train_x_norm, train_y_norm, val_x_norm, val_y_norm, epochs=500)

        print("Training candidate network")
        train_model(candidate, train_x_norm, train_y_norm, val_x_norm, val_y_norm, epochs=100)

        reference_physical = fold_normalization(reference, x_mean, x_std, y_mean, y_std)
        candidate_physical = fold_normalization(candidate, x_mean, x_std, y_mean, y_std)
        save_models(reference_physical, candidate_physical, input_size, output_size)
        print("Saved trained models to", MODEL_FILE)

    reference_weights, reference_biases = extract_weights_and_biases(reference_physical)
    candidate_weights, candidate_biases = extract_weights_and_biases(candidate_physical)

    ref_mse, ref_max_error = evaluate_physical_model(reference_physical, val_x, val_y)
    cand_mse, cand_max_error = evaluate_physical_model(candidate_physical, val_x, val_y)

    print("Reference physical validation mse:", ref_mse)
    print("Reference physical validation max abs error:", ref_max_error)
    print("Candidate physical validation mse:", cand_mse)
    print("Candidate physical validation max abs error:", cand_max_error)
    print("Reference layers:", len(reference_weights))
    print("Candidate layers:", len(candidate_weights))

    print("Running gradient ascent for initial information")
    initial_information, gradient_error = find_initial_information_with_gradient_ascent(
        reference_physical,
        candidate_physical,
        information_lower,
        information_upper,
        restarts=GRADIENT_ASCENT_RESTARTS,
        steps=GRADIENT_ASCENT_STEPS,
        learning_rate=GRADIENT_ASCENT_LR,
    )
    print("Gradient ascent initial error:", gradient_error)

    if HORIZON_SWEEP:
        certifications = []
        for horizon in range(1, CERTIFICATION_HORIZON + 1):
            certifications.append(
                run_certification(
                    reference_weights,
                    reference_biases,
                    candidate_weights,
                    candidate_biases,
                    information_lower,
                    information_upper,
                    initial_information,
                    horizon,
                )
            )
    else:
        certifications = [
            run_certification(
                reference_weights,
                reference_biases,
                candidate_weights,
                candidate_biases,
                information_lower,
                information_upper,
                initial_information,
                CERTIFICATION_HORIZON,
            )
        ]

    final_certification = certifications[-1]
    input_sequence = get_input_sequence_from_certification(final_certification)
    if input_sequence is None:
        print("No MILP solution available, plotting with zero future inputs")
        fallback_steps = T + CERTIFICATION_HORIZON - 1
        input_sequence = np.zeros((fallback_steps, 1), dtype=np.float32)
    else:
        print_input_sequence(input_sequence)

    reference_outputs, candidate_outputs = rollout_reference_candidate(
        reference_physical,
        candidate_physical,
        initial_information,
        input_sequence,
    )
    plot_output_trajectory(
        reference_outputs,
        candidate_outputs,
        final_certification["comparison_start"],
        TRAJECTORY_PLOT_FILE,
    )
    print("Saved output trajectory plot to", TRAJECTORY_PLOT_FILE)
    plot_input_sequence(
        input_sequence,
        final_certification["comparison_start"],
        INPUT_PLOT_FILE,
    )
    print("Saved input sequence plot to", INPUT_PLOT_FILE)

    torch.save(
        {
            "reference": reference_physical.state_dict(),
            "candidate": candidate_physical.state_dict(),
            "reference_weights": reference_weights,
            "reference_biases": reference_biases,
            "candidate_weights": candidate_weights,
            "candidate_biases": candidate_biases,
            "setup": {
                "T": T,
                "certification_horizon": CERTIFICATION_HORIZON,
                "u_min": U_MIN,
                "u_max": U_MAX,
                "x_min": X_MIN,
                "x_max": X_MAX,
                "noise_std": NOISE_STD,
                "input_size": input_size,
                "output_size": output_size,
                "reference_hidden_layers": REFERENCE_HIDDEN_LAYERS,
                "reference_neurons": REFERENCE_NEURONS,
                "candidate_hidden_layers": CANDIDATE_HIDDEN_LAYERS,
                "candidate_neurons": CANDIDATE_NEURONS,
            },
        },
        "trained_models.pt",
    )


if __name__ == "__main__":
    main()

import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import numpy as np
import torch

import main
from milp_certification import solve_certification_milp


N_VALUES = [1, 2, 5, 10, 20]
RESULTS_DIR = "paper_results"
TABLE_FILE = os.path.join(RESULTS_DIR, "certification_table.dat")
SOLVER_BACKEND = "gurobi"
MIP_RELATIVE_GAP = 0.0
TIME_LIMIT = None


def make_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def train_reference_and_candidate(train_x, train_y, val_x, val_y):
    x_mean, x_std, y_mean, y_std = main.compute_normalization(train_x, train_y)
    train_x_norm, train_y_norm = main.normalize_data(train_x, train_y, x_mean, x_std, y_mean, y_std)
    val_x_norm, val_y_norm = main.normalize_data(val_x, val_y, x_mean, x_std, y_mean, y_std)

    input_size = train_x.shape[1]
    output_size = train_y.shape[1]

    reference = main.ReluMlp(
        input_size,
        hidden_layers=main.REFERENCE_HIDDEN_LAYERS,
        neurons_per_layer=main.REFERENCE_NEURONS,
        output_size=output_size,
    )
    candidate = main.ReluMlp(
        input_size,
        hidden_layers=main.CANDIDATE_HIDDEN_LAYERS,
        neurons_per_layer=main.CANDIDATE_NEURONS,
        output_size=output_size,
    )

    print("Training reference network")
    main.train_model(reference, train_x_norm, train_y_norm, val_x_norm, val_y_norm, epochs=500)

    print("Training candidate network")
    main.train_model(candidate, train_x_norm, train_y_norm, val_x_norm, val_y_norm, epochs=100)

    reference_physical = main.fold_normalization(reference, x_mean, x_std, y_mean, y_std)
    candidate_physical = main.fold_normalization(candidate, x_mean, x_std, y_mean, y_std)

    return reference_physical, candidate_physical


def solve_for_horizon(
    reference_weights,
    reference_biases,
    candidate_weights,
    candidate_biases,
    information_lower,
    information_upper,
    initial_information,
    horizon,
):
    print("Solving paper experiment for N =", horizon)
    return solve_certification_milp(
        reference_weights,
        reference_biases,
        candidate_weights,
        candidate_biases,
        horizon=horizon,
        window_length=main.T,
        y_min=main.X_MIN,
        y_max=main.X_MAX,
        u_min=main.U_MIN,
        u_max=main.U_MAX,
        information_lower=information_lower,
        information_upper=information_upper,
        initial_information=initial_information,
        warmup_steps=main.T,
        big_m_value=main.MILP_BIG_M,
        error_bound=main.MILP_P,
        time_limit=TIME_LIMIT,
        num_threads=main.MILP_THREADS,
        mip_relative_gap=MIP_RELATIVE_GAP,
        solver_backend=SOLVER_BACKEND,
        show_solver_output=True,
    )


def save_output_curve_dat(horizon, reference_outputs, candidate_outputs, comparison_start):
    file_name = os.path.join(RESULTS_DIR, "output_curve_N" + str(horizon) + ".dat")
    errors = np.abs(candidate_outputs - reference_outputs)

    with open(file_name, "w", encoding="utf-8") as file:
        file.write("# k y_ref y_candidate abs_error certified\n")
        for k in range(reference_outputs.shape[0]):
            certified = 1 if k >= comparison_start else 0
            file.write(
                str(k)
                + " "
                + str(reference_outputs[k])
                + " "
                + str(candidate_outputs[k])
                + " "
                + str(errors[k])
                + " "
                + str(certified)
                + "\n"
            )

    return file_name


def save_input_curve_dat(horizon, input_sequence, comparison_start):
    file_name = os.path.join(RESULTS_DIR, "input_curve_N" + str(horizon) + ".dat")

    with open(file_name, "w", encoding="utf-8") as file:
        file.write("# k u certified\n")
        for k in range(input_sequence.shape[0]):
            certified = 1 if k >= comparison_start else 0
            file.write(str(k) + " " + str(input_sequence[k, 0]) + " " + str(certified) + "\n")

    return file_name


def write_output_tikz(horizon, data_file):
    tex_file = os.path.join(RESULTS_DIR, "output_curve_N" + str(horizon) + ".tex")
    relative_data_file = os.path.basename(data_file)

    content = r"""\documentclass[tikz,border=2mm]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=12cm,
    height=7cm,
    grid=both,
    xlabel={$k$},
    ylabel={$y$},
    legend pos=north east,
]
\addplot+[mark=o, thick] table[x index=0, y index=1, comment chars={\#}] {""" + relative_data_file + r"""};
\addlegendentry{reference}
\addplot+[mark=square, thick] table[x index=0, y index=2, comment chars={\#}] {""" + relative_data_file + r"""};
\addlegendentry{candidate}
\end{axis}
\end{tikzpicture}
\end{document}
"""

    with open(tex_file, "w", encoding="utf-8") as file:
        file.write(content)

    return tex_file


def write_error_tikz(horizon, data_file):
    tex_file = os.path.join(RESULTS_DIR, "error_curve_N" + str(horizon) + ".tex")
    relative_data_file = os.path.basename(data_file)

    content = r"""\documentclass[tikz,border=2mm]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=12cm,
    height=7cm,
    grid=both,
    xlabel={$k$},
    ylabel={$|y_{\mathrm{cand}}-y_{\mathrm{ref}}|$},
]
\addplot+[mark=diamond, thick] table[x index=0, y index=3, comment chars={\#}] {""" + relative_data_file + r"""};
\end{axis}
\end{tikzpicture}
\end{document}
"""

    with open(tex_file, "w", encoding="utf-8") as file:
        file.write(content)

    return tex_file


def write_input_tikz(horizon, data_file):
    tex_file = os.path.join(RESULTS_DIR, "input_curve_N" + str(horizon) + ".tex")
    relative_data_file = os.path.basename(data_file)

    content = r"""\documentclass[tikz,border=2mm]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=12cm,
    height=6cm,
    grid=both,
    xlabel={$k$},
    ylabel={$u$},
]
\addplot+[const plot, mark=o, thick] table[x index=0, y index=1, comment chars={\#}] {""" + relative_data_file + r"""};
\end{axis}
\end{tikzpicture}
\end{document}
"""

    with open(tex_file, "w", encoding="utf-8") as file:
        file.write(content)

    return tex_file


def save_table(rows):
    with open(TABLE_FILE, "w", encoding="utf-8") as file:
        file.write("# N objective gap_percent solving_time_seconds\n")
        for row in rows:
            file.write(
                str(row["N"])
                + " "
                + str(row["objective"])
                + " "
                + str(row["gap_percent"])
                + " "
                + str(row["runtime"])
                + "\n"
            )


def main_experiment():
    make_results_dir()
    main.set_seed(main.SEED)

    print("Generating dataset")
    train_x, train_y = main.generate_dataset(main.N_TRAIN)
    val_x, val_y = main.generate_dataset(main.N_VAL)
    information_lower, information_upper = main.compute_information_bounds_from_data(
        train_x,
        main.DATA_BOUNDS_MARGIN,
    )

    reference_model, candidate_model = train_reference_and_candidate(train_x, train_y, val_x, val_y)
    reference_weights, reference_biases = main.extract_weights_and_biases(reference_model)
    candidate_weights, candidate_biases = main.extract_weights_and_biases(candidate_model)

    print("Running gradient ascent for initial information")
    initial_information, gradient_error = main.find_initial_information_with_gradient_ascent(
        reference_model,
        candidate_model,
        information_lower,
        information_upper,
        restarts=main.GRADIENT_ASCENT_RESTARTS,
        steps=main.GRADIENT_ASCENT_STEPS,
        learning_rate=main.GRADIENT_ASCENT_LR,
    )
    print("Gradient ascent initial error:", gradient_error)

    table_rows = []

    for horizon in N_VALUES:
        certification = solve_for_horizon(
            reference_weights,
            reference_biases,
            candidate_weights,
            candidate_biases,
            information_lower,
            information_upper,
            initial_information,
            horizon,
        )

        result = certification["result"]
        input_sequence = main.get_input_sequence_from_certification(certification)

        if input_sequence is None:
            raise RuntimeError("Gurobi did not return a feasible solution for N=" + str(horizon))

        reference_outputs, candidate_outputs = main.rollout_reference_candidate(
            reference_model,
            candidate_model,
            initial_information,
            input_sequence,
        )

        output_dat = save_output_curve_dat(
            horizon,
            reference_outputs,
            candidate_outputs,
            certification["comparison_start"],
        )
        input_dat = save_input_curve_dat(
            horizon,
            input_sequence,
            certification["comparison_start"],
        )

        write_output_tikz(horizon, output_dat)
        write_error_tikz(horizon, output_dat)
        write_input_tikz(horizon, input_dat)

        if result["success"]:
            gap_percent = 0.0
        else:
            gap = result["mip_gap"]
            if gap is None:
                gap_percent = None
            else:
                gap_percent = 100.0 * gap

        table_rows.append(
            {
                "N": horizon,
                "objective": certification["objective_value"],
                "gap_percent": gap_percent,
                "runtime": result["runtime"],
            }
        )

        print(
            "N =",
            horizon,
            "objective =",
            certification["objective_value"],
            "gap =",
            gap_percent,
            "runtime =",
            result["runtime"],
        )

    save_table(table_rows)
    print("Saved table to", TABLE_FILE)
    print("Saved curves and TikZ files to", RESULTS_DIR)


if __name__ == "__main__":
    main_experiment()

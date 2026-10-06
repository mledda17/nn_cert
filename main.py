import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from milp_certification import solve_certification_milp


# Numerical setup from the paper.
T = 5
CERTIFICATION_HORIZON = 1
MILP_TIME_LIMIT = 600.0
MILP_THREADS = 8
DATA_BOUNDS_MARGIN = 0.05
N_TRAIN = 20_000
N_VAL = 2_000
NOISE_STD = 0.02

U_MIN = -2.0
U_MAX = 2.0
X_MIN = 0.0
X_MAX = 10.0

K1 = 0.5
K2 = 0.4
K3 = 0.2
K4 = 0.3

SEED = 1


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

    reference = ReluMlp(input_size, hidden_layers=1, neurons_per_layer=16, output_size=output_size)
    candidate = ReluMlp(input_size, hidden_layers=1, neurons_per_layer=8, output_size=output_size)

    print("Reference parameters:", count_parameters(reference))
    print("Candidate parameters:", count_parameters(candidate))

    print("Training reference network")
    train_model(reference, train_x_norm, train_y_norm, val_x_norm, val_y_norm, epochs=500)

    print("Training candidate network")
    train_model(candidate, train_x_norm, train_y_norm, val_x_norm, val_y_norm, epochs=100)

    reference_physical = fold_normalization(reference, x_mean, x_std, y_mean, y_std)
    candidate_physical = fold_normalization(candidate, x_mean, x_std, y_mean, y_std)

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

    print("Solving certification MILP")
    certification = solve_certification_milp(
        reference_weights,
        reference_biases,
        candidate_weights,
        candidate_biases,
        horizon=CERTIFICATION_HORIZON,
        window_length=T,
        y_min=X_MIN,
        y_max=X_MAX,
        u_min=U_MIN,
        u_max=U_MAX,
        information_lower=information_lower,
        information_upper=information_upper,
        time_limit=MILP_TIME_LIMIT,
        num_threads=MILP_THREADS,
        show_solver_output=True,
    )
    milp_result = certification["result"]
    print("MILP success:", milp_result.success)
    print("MILP status:", milp_result.status)
    print("MILP message:", milp_result.message)
    print("Certified objective:", certification["objective_value"])
    print("MILP error bound P:", certification["error_bound"])

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
            },
        },
        "trained_models.pt",
    )


if __name__ == "__main__":
    main()

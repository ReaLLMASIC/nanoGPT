import argparse
from datetime import datetime
import json
import os
import subprocess

import torch
from vizier.service import clients, pyvizier as vz
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vizier optimization based on json configuration file."
    )
    parser.add_argument("--output_dir", type=str, default="out", help="Directory to place the set of output checkpoints.")
    parser.add_argument("--vizier_iterations", type=int, default=20, help="Number of Vizier iterations.")
    parser.add_argument("--vizier_algorithm", choices=[
        "GAUSSIAN_PROCESS_BANDIT",
        "RANDOM_SEARCH",
        "QUASI_RANDOM_SEARCH",
        "GRID_SEARCH",
    ], default="GAUSSIAN_PROCESS_BANDIT", help="Choose the Vizier algorithm to use.")
    return parser.parse_args()

def get_best_val_loss(out_dir):
    best_val_loss_file = os.path.join(out_dir, "best_val_loss_and_iter.txt")
    if os.path.exists(best_val_loss_file):
        with open(best_val_loss_file, "r") as file:
            try:
                best_val_loss = float(file.readline().strip().split(",")[0])
                return best_val_loss
            except ValueError:
                print("val_loss file not found, trying checkpoint...")
    checkpoint_file = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    best_val_loss = checkpoint["best_val_loss"]
    return best_val_loss

def factorize_matrix(A, output_dir):
    n_rows = 50000
    n_cols = 384
    original_matrix = torch.randn(n_rows, n_cols)

    W1 = torch.randn(n_rows, A, requires_grad=True)
    W2 = torch.randn(A, n_cols, requires_grad=True)

    optimizer = optim.Adam([W1, W2], lr=1e-3)
    loss_fn = nn.MSELoss()

    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        reconstructed_matrix = torch.matmul(W1, W2)
        loss = loss_fn(reconstructed_matrix, original_matrix)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    os.makedirs(output_dir, exist_ok=True)
    torch.save({"best_val_loss": loss.item()}, os.path.join(output_dir, "ckpt.pt"))
    with open(os.path.join(output_dir, "best_val_loss_and_iter.txt"), "w") as file:
        file.write(f"{loss.item()},{epoch+1}\n")
    return loss.item()

def run_experiment_with_vizier(output_dir, vizier_algorithm, vizier_iterations):
    search_space = vz.SearchSpace()
    search_space.root.add_int_param(name="A", min_value=10, max_value=500)

    study_config = vz.StudyConfig(
        search_space=search_space,
        metric_information=[
            vz.MetricInformation(name="loss", goal=vz.ObjectiveMetricGoal.MINIMIZE)
        ],
    )
    study_config.algorithm = vizier_algorithm
    study_client = clients.Study.from_study_config(
        study_config, owner="owner", study_id="example_study_id"
    )

    for i in range(vizier_iterations):
        print("Vizier Iteration", i)
        suggestions = study_client.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            A = params["A"]
            loss = factorize_matrix(A, os.path.join(output_dir, f"run_{i}_A_{A}"))
            suggestion.complete(vz.Measurement(metrics={"loss": loss}))

    optimal_trials = study_client.optimal_trials()
    for trial in optimal_trials:
        best_trial = trial.materialize()
        print(f"Best trial: {best_trial.parameters}, Loss: {best_trial.final_measurement.metrics['loss']}")

def main():
    args = parse_args()
    run_experiment_with_vizier(args.output_dir, args.vizier_algorithm, args.vizier_iterations)

if __name__ == "__main__":
    main()


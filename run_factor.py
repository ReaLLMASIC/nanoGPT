import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import pandas as pd
from statistics import mean, median, stdev
from vizier.service import clients, pyvizier as vz
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

# Factorization function with configurable A and seed
def factorize_matrix(A, original_matrix, device, num_epochs, seed, progress, task_id):
    A = int(A)  # Ensure A is an integer
    torch.manual_seed(seed)
    n_rows, n_cols = original_matrix.shape

    W1 = torch.randn((n_rows, A), requires_grad=True, device=device)
    W2 = torch.randn((A, n_cols), requires_grad=True, device=device)

    optimizer = optim.Adam([W1, W2], lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        reconstructed_matrix = torch.matmul(W1, W2)
        loss = loss_fn(reconstructed_matrix, original_matrix)
        loss.backward()
        optimizer.step()
        
        progress.update(task_id, advance=1, description=f"Loss: {loss.item():.4f}")
        
    return loss.item()

def run_experiment_with_vizier(vizier_algorithm, vizier_iterations, A_start, A_end, num_epochs, num_seeds, original_matrix, device, output_csv):
    search_space = vz.SearchSpace()
    search_space.root.add_int_param(name="A", min_value=A_start, max_value=A_end)

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

    results = []
    console = Console()

    for i in range(vizier_iterations):
        print("Vizier Iteration", i)
        suggestions = study_client.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            A = params["A"]
            seed_losses = []
            for seed in range(num_seeds):
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.1f}%",
                    refresh_per_second=1,
                ) as progress:
                    task_id = progress.add_task(f"Training (Seed {seed+1})", total=num_epochs)
                    loss = factorize_matrix(A, original_matrix, device, num_epochs, seed, progress, task_id)
                seed_losses.append(loss)
            best_loss = min(seed_losses)
            results.append({
                "A": A, 
                "min": min(seed_losses), 
                "max": max(seed_losses), 
                "mean": mean(seed_losses), 
                "median": median(seed_losses), 
                "std": stdev(seed_losses) if len(seed_losses) > 1 else 0
            })
            suggestion.complete(vz.Measurement(metrics={"loss": best_loss}))

        # Print the results table at the end of each iteration
        results_sorted = sorted(results, key=lambda x: x["A"])
        table = Table(title=f"Results after Vizier Iteration {i + 1}")
        table.add_column("A", justify="right", style="cyan")
        table.add_column("Min Loss", justify="right", style="magenta")
        table.add_column("Max Loss", justify="right", style="magenta")
        table.add_column("Mean Loss", justify="right", style="magenta")
        table.add_column("Median Loss", justify="right", style="magenta")
        table.add_column("Std Dev", justify="right", style="magenta")

        for result in results_sorted:
            table.add_row(
                str(result["A"]), 
                f"{result['min']:.4f}", 
                f"{result['max']:.4f}", 
                f"{result['mean']:.4f}", 
                f"{result['median']:.4f}", 
                f"{result['std']:.4f}"
            )

        console.clear()
        console.print(table)

        # Save results to CSV
        df = pd.DataFrame(results_sorted)
        df.to_csv(output_csv, index=False)

    # Print the final sorted results
    console.clear()
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Run matrix factorization with Vizier optimization.")
    parser.add_argument('--vizier_algorithm', type=str, choices=[
        "GP_UCB_PE", "GAUSSIAN_PROCESS_BANDIT", "RANDOM_SEARCH", "QUASI_RANDOM_SEARCH",
        "GRID_SEARCH", "SHUFFLED_GRID_SEARCH", "EAGLE_STRATEGY", "CMA_ES",
        "EMUKIT_GP_EI", "NSGA2", "BOCS", "HARMONICA"
    ], default="GAUSSIAN_PROCESS_BANDIT", help="Choose the Vizier algorithm to use.")
    parser.add_argument('--vizier_iterations', type=int, default=20, help="Number of Vizier iterations.")
    parser.add_argument('--num_epochs', type=int, default=1000, help="Number of training epochs.")
    parser.add_argument('--num_seeds', type=int, default=5, help="Number of random seeds for each A value.")
    parser.add_argument('--A_start', type=int, default=10, help="Minimum value of A for optimization.")
    parser.add_argument('--A_end', type=int, default=100, help="Maximum value of A for optimization.")
    parser.add_argument('--output_csv', type=str, default="results.csv", help="Path to the output CSV file.")
    parser.add_argument('--matrix_path', type=str, default=None, help="Path to the matrix .npy file for factorization.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.matrix_path:
        original_matrix = torch.from_numpy(np.load(args.matrix_path)).to(device)
    else:
        original_matrix = torch.randn(50000, 384).to(device)

    run_experiment_with_vizier(
        args.vizier_algorithm,
        args.vizier_iterations,
        args.A_start,
        args.A_end,
        args.num_epochs,
        args.num_seeds,
        original_matrix,
        device,
        args.output_csv
    )

if __name__ == "__main__":
    main()


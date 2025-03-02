import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# Learning rate decay functions
def cosine_decay_lr(epoch, total_epochs, lr_start, lr_stop):
    return lr_stop + 0.5 * (lr_start - lr_stop) * (1 + np.cos(np.pi * epoch / total_epochs))

def linear_decay_lr(epoch, total_epochs, lr_start, lr_stop):
    return lr_start - (lr_start - lr_stop) * (epoch / total_epochs)

# Custom loss function for direction and magnitude alignment
def direction_magnitude_loss(reconstructed_matrix, original_matrix):
    cosine_loss = 1 - F.cosine_similarity(reconstructed_matrix, original_matrix, dim=1).mean()
    norm_loss = F.mse_loss(reconstructed_matrix.norm(dim=1), original_matrix.norm(dim=1))
    return cosine_loss + norm_loss

# Factorization function with configurable A, seed, and loss function
def factorize_matrix(A, original_matrix, device, num_epochs, seed, progress, task_id, lr_start, lr_stop, lr_decay, output_dir=None, loss_fn='mse'):
    A = int(A)  # Ensure A is an integer
    torch.manual_seed(seed)
    n_rows, n_cols = original_matrix.shape

    W1 = torch.randn((n_rows, A), requires_grad=True, device=device)
    W2 = torch.randn((A, n_cols), requires_grad=True, device=device)

    optimizer = optim.Adam([W1, W2], lr=lr_start)

    # Choose the loss function
    if loss_fn == 'mse':
        loss_fn_obj = nn.MSELoss()
    elif loss_fn == 'mae':
        loss_fn_obj = nn.L1Loss()
    elif loss_fn == 'huber':
        loss_fn_obj = nn.SmoothL1Loss()
    elif loss_fn == 'cosine':
        loss_fn_obj = lambda pred, target: 1 - F.cosine_similarity(pred.flatten(), target.flatten(), dim=0).mean()
    elif loss_fn == 'frobenius':
        loss_fn_obj = lambda pred, target: torch.norm(pred - target, p='fro')
    elif loss_fn == 'direction_magnitude':
        loss_fn_obj = direction_magnitude_loss

    best_W1, best_W2, best_loss = None, None, float('inf')

    for epoch in range(num_epochs):
        if lr_decay == 'cosine':
            lr = cosine_decay_lr(epoch, num_epochs, lr_start, lr_stop)
        elif lr_decay == 'linear':
            lr = linear_decay_lr(epoch, num_epochs, lr_start, lr_stop)
        else:
            lr = lr_start  # No decay

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        reconstructed_matrix = torch.matmul(W1, W2)
        loss = loss_fn_obj(reconstructed_matrix, original_matrix)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_W1, best_W2 = W1.detach().cpu().numpy(), W2.detach().cpu().numpy()

        progress.update(task_id, advance=1, description=f"Loss: {loss.item():.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        wte_file_path = os.path.join(output_dir, f"{A}_{seed}_wte.npy")
        scale_matrices_file_path = os.path.join(output_dir, f"{A}_{seed}_scale_matrices.npz")
        np.save(wte_file_path, best_W1)
        np.savez(scale_matrices_file_path, scale_up=best_W2, scale_down=best_W2)
        print(f"Saved W1 to {wte_file_path} and W2 to {scale_matrices_file_path}")

    return best_loss, best_W1, best_W2

def run_experiment_with_vizier(viz_owner, viz_studyid, vizier_algorithm, vizier_iterations, A_start, A_step, A_end, num_epochs, num_seeds, original_matrix, device, output_csv, output_dir, loss_fn, lr_start, lr_stop, lr_decay):
    search_space = vz.SearchSpace()
    feasible_values_list = []
    if A_step == 1:
        search_space.root.add_int_param(name="A", min_value=A_start, max_value=A_end)
    else:
        feasible_values_list = list(range(A_start, A_end, A_step))
        search_space.root.add_discrete_param(name="A", feasible_values=feasible_values_list)

    study_config = vz.StudyConfig(
        search_space=search_space,
        metric_information=[
            vz.MetricInformation(name="loss", goal=vz.ObjectiveMetricGoal.MINIMIZE)
        ],
    )
    study_config.algorithm = vizier_algorithm
    study_client = clients.Study.from_study_config(
        study_config, owner=viz_owner, study_id=viz_studyid
    )

    results = []
    console = Console()

    original_size = original_matrix.numel()
    n_rows, n_cols = original_matrix.shape

    if vizier_iterations is None:
        if A_step == 1:
            vizier_iterations = A_end - A_start + 1
        else:
            vizier_iterations = len(feasible_values_list)

    for i in range(vizier_iterations):
        print("Vizier Iteration", i)
        suggestions = study_client.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            A = params["A"]
            seed_losses = []
            best_seed_loss = float('inf')
            best_W1, best_W2 = None, None

            for seed in range(num_seeds):
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.1f}%",
                    refresh_per_second=1,
                ) as progress:
                    task_id = progress.add_task(f"Training (Seed {seed+1})", total=num_epochs)
                    loss, W1, W2 = factorize_matrix(A, original_matrix, device, num_epochs, seed, progress, task_id, lr_start, lr_stop, lr_decay, output_dir, loss_fn)
                seed_losses.append(loss)

                if loss < best_seed_loss:
                    best_seed_loss = loss
                    best_W1, best_W2 = W1, W2

            # Save the best factorization for this A
            best_wte_file_path = os.path.join(output_dir, f"{A}_best_wte.npy")
            best_scale_matrices_file_path = os.path.join(output_dir, f"{A}_best_scale_matrices.npz")
            np.save(best_wte_file_path, best_W1)
            np.savez(best_scale_matrices_file_path, scale_up=best_W2, scale_down=best_W2)
            print(f"Saved best W1 for A={A} to {best_wte_file_path} and best W2 to {best_scale_matrices_file_path}")

            best_loss = min(seed_losses)
            W1_size = best_W1.size
            W2_size = best_W2.size
            compression_ratio = original_size / (W1_size + W2_size)

            results.append({
                "A": A,
                "min": min(seed_losses),
                "max": max(seed_losses),
                "mean": mean(seed_losses),
                "median": median(seed_losses),
                "std": stdev(seed_losses) if len(seed_losses) > 1 else 0,
                "num_seeds": num_seeds,
                "loss_fn": loss_fn,
                "W1_shape": best_W1.shape,
                "W2_shape": best_W2.shape,
                "original_shape": original_matrix.shape,
                "compression_ratio": compression_ratio,
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
        table.add_column("Num Seeds", justify="right", style="blue")
        table.add_column("Loss Fn", justify="right", style="green")
        table.add_column("W1 Shape", justify="right", style="yellow")
        table.add_column("W2 Shape", justify="right", style="yellow")
        table.add_column("Orig Shape", justify="right", style="yellow")
        table.add_column("Param Ratio", justify="right", style="red")

        for result in results_sorted:
            table.add_row(
                str(result["A"]),
                f"{result['min']:.4f}",
                f"{result['max']:.4f}",
                f"{result['mean']:.4f}",
                f"{result['median']:.4f}",
                f"{result['std']:.4f}",
                str(result["num_seeds"]),
                result["loss_fn"],
                str(result["W1_shape"]),
                str(result["W2_shape"]),
                "(" + str(result["original_shape"][0]) + ", " +
                str(result["original_shape"][1]) + ")",
                f"{result['compression_ratio']:.4f}",
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
    ], default="GRID_SEARCH", help="Choose the Vizier algorithm to use.")
    parser.add_argument('--viz_owner', type=str, default="viz_owner", help="vizier owner")
    parser.add_argument('--viz_studyid', type=str, default="0", help="study_id")
    parser.add_argument('--vizier_iterations', type=int, default=None, help="Number of Vizier iterations.")
    parser.add_argument('--num_epochs', type=int, default=2000, help="Number of training epochs.")
    parser.add_argument('--num_seeds', type=int, default=5, help="Number of random seeds for each A value.")
    parser.add_argument('--A_start', type=int, default=10, help="Minimum value of A for optimization.")
    parser.add_argument('--A_step', type=int, default=5, help="Step between start and end.")
    parser.add_argument('--A_end', type=int, default=1000, help="Maximum value of A for optimization.")
    parser.add_argument('--output_csv', type=str, default="results.csv", help="Path to the output CSV file.")
    parser.add_argument('--output_dir', type=str, default="out", help="Directory to save output factorization results.")
    parser.add_argument('--matrix_path', type=str, default=None, help="Path to the matrix .npy file for factorization.")
    parser.add_argument('--loss_fn', type=str, choices=['mse', 'mae', 'huber', 'cosine', 'frobenius', 'direction_magnitude'], default='direction_magnitude', help="Loss function to use for matrix approximation.")
    parser.add_argument('--lr_start', type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument('--lr_stop', type=float, default=1e-5, help="Final learning rate for decay.")
    parser.add_argument('--lr_decay', type=str, choices=['none', 'cosine', 'linear'], default='cosine', help="Learning rate decay method.")
    parser.add_argument('--random_init_matrix', action='store_true', help="Use randomly initialized matrix with specified mean and stddev.")
    parser.add_argument('--random_matrix_vocab', type=float, default=50257, help="Random matrix vocab dimension (default tiktoken 50257).")
    parser.add_argument('--random_matrix_n_embd', type=float, default=768, help="Random matrix n_embd dimension (default 768).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.matrix_path:
        original_matrix = torch.from_numpy(np.load(args.matrix_path)).to(device)
    else:
        original_matrix = torch.randn(args.random_matrix_vocab, ags.random_matrix_n_embd).to(device)

    run_experiment_with_vizier(
        args.viz_owner,
        args.viz_studyid,
        args.vizier_algorithm,
        args.vizier_iterations,
        args.A_start,
        args.A_step,
        args.A_end,
        args.num_epochs,
        args.num_seeds,
        original_matrix,
        device,
        args.output_csv,
        args.output_dir,
        args.loss_fn,
        args.lr_start,
        args.lr_stop,
        args.lr_decay
    )

if __name__ == "__main__":
    main()


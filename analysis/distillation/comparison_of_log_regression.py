import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os
from tqdm import tqdm  # Import tqdm for progress bars

# Define function for logarithmic fit
def log_func(x, a, b):
    return a * np.log1p(x) + b

# Define function for exponential decay fit
def exp_decay_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Define MSE Inverse (MSEI) metric
def mse_inverse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return 1 / (1 + mse)  # Higher MSEI means better fit

def process_embedding_dims(min_pow, max_pow, regression_type, num_vectors, mean, stddev):
    # Generate embedding dimensions
    embedding_dims = [2 ** i for i in range(min_pow, max_pow + 1)]

    # Store regression trends for final comparison plot
    regression_trends = {}

    # Outer progress bar for embedding dimensions
    for dim in tqdm(embedding_dims, desc="Processing Dimensions", unit="dim"):
        data_filename = f"angle_distribution_{dim}d.npy"

        if os.path.exists(data_filename):
            print(f"\nLoading existing data file: {data_filename}")
            data_dict = np.load(data_filename, allow_pickle=True).item()
        else:
            print(f"\nGenerating new data file: {data_filename}")
            np.random.seed(42)  # For reproducibility
            vectors = np.random.normal(mean, stddev, size=(1, dim))  # First vector

            min_angles = []

            # Inner progress bar for vector generation
            for i in tqdm(range(1, num_vectors + 1), desc=f"{dim}D Vectors", unit="vec", leave=False):
                new_vector = np.random.normal(mean, stddev, size=(dim,))

                # Compute cosine similarity with **all** prior vectors
                cos_sim = np.dot(vectors, new_vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(new_vector))
                # Convert cosine similarity to angles (radians to degrees)
                angles = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
                # Store minimum angle
                min_angles.append(np.min(angles))
                # Append new vector to full history
                vectors = np.vstack((vectors, new_vector))

            # Save raw data
            data_dict = {"x_vals": np.arange(1, num_vectors + 1), "min_angles": np.array(min_angles)}
            np.save(data_filename, data_dict)

        x_vals = data_dict["x_vals"]
        min_angles = data_dict["min_angles"]

        # Fit selected regression models
        fit_results = {}
        if regression_type in ["log", "both"]:
            popt_log, _ = curve_fit(log_func, x_vals, min_angles)
            trend_min_log = log_func(x_vals, *popt_log)
            msei_log = mse_inverse(np.array(min_angles), trend_min_log)
            fit_results["log"] = (trend_min_log, msei_log)
            regression_trends[f"{dim}D Log"] = (x_vals, min_angles, trend_min_log, msei_log)

        if regression_type in ["exp", "both"]:
            popt_exp, _ = curve_fit(exp_decay_func, x_vals, min_angles, p0=[5, 0.001, 80])
            trend_min_exp = exp_decay_func(x_vals, *popt_exp)
            msei_exp = mse_inverse(np.array(min_angles), trend_min_exp)
            fit_results["exp"] = (trend_min_exp, msei_exp)
            regression_trends[f"{dim}D Exp"] = (x_vals, min_angles, trend_min_exp, msei_exp)

        # Save individual plots
        plt.figure(figsize=(10, 6))
        plt.scatter(x_vals, min_angles, color="red", s=1, alpha=0.3, label="Min Angle")
        plt.xlabel("Number of Vectors Added")
        plt.ylabel("Angle (Degrees)")
        plt.title(f"Minimum Angle Distribution ({dim}-Dim)")
        plt.legend()
        plt.savefig(f"angle_distribution_{dim}d_min_angle.png", dpi=300)
        plt.close()

        # Overlay plot with points and regression fits
        plt.figure(figsize=(10, 6))
        plt.scatter(x_vals, min_angles, color="red", s=1, alpha=0.3, label="Min Angle")
        if "log" in fit_results:
            plt.plot(x_vals, fit_results["log"][0], color="black", linestyle="--", linewidth=2,
                     label=f"Log Fit (MSEI={fit_results['log'][1]:.4f})")
        if "exp" in fit_results:
            plt.plot(x_vals, fit_results["exp"][0], color="blue", linestyle="-.", linewidth=2,
                     label=f"Exp Fit (MSEI={fit_results['exp'][1]:.4f})")
        plt.xlabel("Number of Vectors Added")
        plt.ylabel("Angle (Degrees)")
        plt.title(f"Regression Fit for Min Angle ({dim}-Dim)")
        plt.legend()
        plt.savefig(f"angle_distribution_{dim}d_regression.png", dpi=300)
        plt.close()

        print(f"Saved plots for {dim}D")

    # Generate final comparison chart for all regression fits
    plt.figure(figsize=(12, 8))
    for label, (x_vals, min_angles, trend, msei) in regression_trends.items():
        color = next(plt.gca()._get_lines.prop_cycler)["color"]
        plt.scatter(x_vals, min_angles, color=color, s=1, alpha=0.3,
                    label=label.replace("Log", "Min Angle").replace("Exp", "Min Angle"))
        linestyle = "--" if "Log" in label else "-."
        plt.plot(x_vals, trend, color=color, linestyle=linestyle, linewidth=2,
                 label=f"{label.replace('Min Log', 'Log Fit').replace('Min Exp', 'Exp Fit')} (MSEI={msei:.4f})")

    plt.xlabel("Number of Vectors Added")
    plt.ylabel("Angle (Degrees)")
    plt.title("Comparison of Regression Models Across Dimensions")
    plt.legend()

    # Save comparison plot
    comparison_plot_filename = f"angle_distribution_{regression_type}_comparison.png"
    plt.savefig(comparison_plot_filename, dpi=300)
    plt.close()

    print(f"Saved: {comparison_plot_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process embedding dimensions for regression analysis.")
    parser.add_argument("--min_pow", type=int, default=2, help="Minimum power of 2 for embedding dims (default: 2)")
    parser.add_argument("--max_pow", type=int, default=10, help="Maximum power of 2 for embedding dims (default: 10)")
    parser.add_argument("--regression", choices=["log", "exp", "both"], default="both",
                        help="Choose regression type: log, exp, or both (default: both)")
    parser.add_argument("--num_vectors", type=int, default=5000, help="Number of vectors (default: 5000)")
    parser.add_argument("--mean", type=float, default=0.0, help="Mean for Gaussian initialization (default: 0.0)")
    parser.add_argument("--stddev", type=float, default=0.02, help="Stddev for Gaussian initialization (default: 0.02)")

    args = parser.parse_args()

    process_embedding_dims(args.min_pow, args.max_pow, args.regression, args.num_vectors, args.mean, args.stddev)


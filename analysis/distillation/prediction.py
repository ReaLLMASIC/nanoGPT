#!/usr/bin/env python3
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def a_pred(D):
    """
    Returns the 'a' parameter from the global power-law fit.
    a(D) = -16 * (D ^ -0.489)
    """
    return -16.0 * (D ** -0.489)

def b_pred(D):
    """
    Returns the 'b' parameter from the global power-law fit.
    b(D) = 90 - 72.5 * (D ^ -0.517)
    """
    return 90.0 - 72.5 * (D ** -0.517)

def main():
    parser = argparse.ArgumentParser(description="Compare CSV-based log-fit values to global power-law fits.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file (with header).")
    args = parser.parse_args()

    dims_list = []
    a_data_list = []
    b_data_list = []

    # Read the CSV, select only rows where Regression Type is "log"
    with open(args.csv, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["Regression Type"] == "log":
                # Grab dimension, a, b from CSV
                # Convert to float while handling the empty 'c'
                D = float(row["Dimension"])
                a_val = float(row["a"])  # from the CSV
                b_val = float(row["b"])  # from the CSV

                dims_list.append(D)
                a_data_list.append(a_val)
                b_data_list.append(b_val)

    # Convert to numpy arrays for easier manipulation
    dims = np.array(dims_list)
    a_data = np.array(a_data_list)
    b_data = np.array(b_data_list)

    # Compute predicted values from the discovered power-law formula
    a_pred_vals = a_pred(dims)
    b_pred_vals = b_pred(dims)

    # Compute a simple Mean Squared Error (MSE) as a measure of fit
    mse_a = np.mean((a_data - a_pred_vals)**2)
    mse_b = np.mean((b_data - b_pred_vals)**2)
    print(f"MSE for a(D) compared to global power-law fit: {mse_a:.6f}")
    print(f"MSE for b(D) compared to global power-law fit: {mse_b:.6f}")

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Subplot for a(D) ---
    axes[0].scatter(dims, a_data, color="red", label="CSV-based a(D)", zorder=3)
    axes[0].plot(dims, a_pred_vals, color="blue", label="Global Fit a(D)", zorder=2)
    axes[0].set_xscale("log")  # optional, since dims are often powers of 2
    axes[0].set_xlabel("Dimension (D)")
    axes[0].set_ylabel("a(D)")
    axes[0].set_title("CSV vs. Global Fit for a(D)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Subplot for b(D) ---
    axes[1].scatter(dims, b_data, color="red", label="CSV-based b(D)", zorder=3)
    axes[1].plot(dims, b_pred_vals, color="blue", label="Global Fit b(D)", zorder=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Dimension (D)")
    axes[1].set_ylabel("b(D)")
    axes[1].set_title("CSV vs. Global Fit for b(D)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


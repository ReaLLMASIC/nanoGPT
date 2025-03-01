import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse

# --------------------------------------------------------------------
# 1. Measure Crowding: compute the minimum angle distribution 
#    after each vector is added, using radians internally.
# --------------------------------------------------------------------
def compute_min_angle_distribution(vectors: torch.Tensor):
    """
    Given a set of vectors of shape [N, D], compute the minimum angle (in radians)
    after each vector is introduced.

    Returns:
        x_vals: array of shape (N,) with values [1..N]
        min_angles_rad: array of shape (N,) with the min angle in radians
                        after each addition
    """
    num_vectors = vectors.shape[0]
    min_angles_rad = []

    # The first vector has no angles to compare to, so set min angle to 0.0
    existing_vectors = vectors[0].unsqueeze(0)
    min_angles_rad.append(0.0)

    for i in range(1, num_vectors):
        new_vector = vectors[i].unsqueeze(0)
        dot_products = (existing_vectors * new_vector).sum(dim=1)
        norms_existing = existing_vectors.norm(dim=1)
        norm_new = new_vector.norm(dim=1)

        cos_sim = dot_products / (norms_existing * norm_new + 1e-9)
        # Angles in radians
        angles_rad = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))

        min_angle = angles_rad.min().item()
        min_angles_rad.append(min_angle)

        # Add the new vector to the existing set
        existing_vectors = torch.cat([existing_vectors, new_vector], dim=0)

    x_vals = np.arange(1, num_vectors + 1)
    return x_vals, np.array(min_angles_rad)


# --------------------------------------------------------------------
# 2. Optimization: push angles between all pairs of vectors toward 90°
#    (π/2 radians) by working in radians, with progress bar and timing.
# --------------------------------------------------------------------
def optimize_angles(vectors: torch.Tensor, iterations=100, lr=0.01, print_frequency=10):
    """
    Optimizes the angles between vectors to be as close to 90 degrees (π/2 radians) 
    as possible, with progress bar and time estimates.

    Args:
        vectors: Tensor of shape [N, D].
        iterations: Number of optimizer steps.
        lr: Learning rate.
        print_frequency: How often to print detailed time/iteration info.

    Returns:
        optimized_vectors: Tensor of shape [N, D], after angle optimization.
    """
    # Make a copy that requires grad
    vectors = vectors.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([vectors], lr=lr)

    # We'll compute loss = Σᵢⱼ (angleᵢⱼ - π/2)² for all i<j.
    half_pi = np.pi / 2.0  # 1.570796...

    start_time = time.time()

    with tqdm(total=iterations, desc="Optimizing angles", unit="iter") as pbar:
        for iteration in range(iterations):
            iteration_start = time.time()

            optimizer.zero_grad()
            loss = 0.0

            N = vectors.size(0)
            for i in range(N):
                for j in range(i + 1, N):
                    dot_product = torch.dot(vectors[i], vectors[j])
                    magnitudes = torch.norm(vectors[i]) * torch.norm(vectors[j])
                    if magnitudes > 0:
                        cosine_angle = torch.clamp(dot_product / magnitudes, -1.0, 1.0)
                        angle_rad = torch.acos(cosine_angle)
                        # Deviation from π/2
                        loss += (angle_rad - half_pi) ** 2
                    else:
                        # If magnitude is zero, encourage them to differ
                        loss += torch.sum((vectors[i] - vectors[j]) ** 2)

            loss.backward()
            optimizer.step()

            # Normalize vectors after each step
            with torch.no_grad():
                norms = torch.norm(vectors.data, dim=1, keepdim=True)
                vectors.data = vectors.data / (norms + 1e-9)

            iteration_end = time.time()
            iteration_time = iteration_end - iteration_start

            # Update progress bar display
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "time/iter (s)": f"{iteration_time:.4f}"
            })
            pbar.update(1)

            # Print more detailed info every `print_frequency` steps
            if (iteration + 1) % print_frequency == 0:
                elapsed = iteration_end - start_time
                per_iter_avg = elapsed / (iteration + 1)
                remaining = iterations - (iteration + 1)
                etc_seconds = per_iter_avg * remaining
                print(
                    f"Iteration [{iteration+1}/{iterations}] "
                    f"Loss={loss.item():.4f}, "
                    f"Avg Time/Iter={per_iter_avg:.4f}s, "
                    f"ETC={etc_seconds:.2f}s"
                )

    return vectors.detach()


# --------------------------------------------------------------------
# 3. Main script: tie everything together
# --------------------------------------------------------------------
def main(args):
    # Create an output directory if needed
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Decide on device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    for dim in args.dimensions:
        print(f"\n=== Dimension {dim} ===")
        # 1) Generate random vectors
        vectors = torch.normal(
            mean=0.0,
            std=0.02,
            size=(args.num_vectors, dim),
            device=device
        )

        # 2) Measure the crowding (min-angle distribution) before optimization (radians)
        x_vals_before, min_angles_before_rad = compute_min_angle_distribution(vectors)

        # 3) Optimize angles
        optimized_vectors = optimize_angles(
            vectors,
            iterations=args.iterations,
            lr=args.lr,
            print_frequency=args.print_freq
        )

        # 4) Measure the crowding (min-angle distribution) after optimization (radians)
        x_vals_after, min_angles_after_rad = compute_min_angle_distribution(optimized_vectors)

        # Convert radians to degrees for plotting
        min_angles_before_deg = np.degrees(min_angles_before_rad)
        min_angles_after_deg = np.degrees(min_angles_after_rad)

        # 5) Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(
            x_vals_before,
            min_angles_before_deg,
            label="Before Optimization",
            color="red",
            alpha=0.8,
        )
        plt.plot(
            x_vals_after,
            min_angles_after_deg,
            label="After Optimization",
            color="blue",
            alpha=0.8,
        )
        plt.xlabel("Number of Vectors Added", fontsize=12)
        plt.ylabel("Minimum Angle (Degrees)", fontsize=12)
        plt.title(f"Min Angle Distribution (Dimension={dim})", fontsize=14)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"min_angle_dim{dim}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and optimize vector angles across different dimensions (radians internally)."
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[2, 4, 16, 32],
        help="List of vector dimensions to analyze."
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1000,
        help="Number of vectors to generate for each dimension."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of optimization steps."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for the angle optimization."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force use of CPU instead of GPU."
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="How often to print the extended time/iteration info."
    )
    args = parser.parse_args()

    main(args)


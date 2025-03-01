import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import os
import numpy as np

def calculate_angle(v1, v2):
    """Calculates the angle between two vectors in degrees."""
    dot_product = torch.dot(v1, v2)
    magnitudes = torch.norm(v1) * torch.norm(v2)

    if magnitudes == 0:
        return 90.0

    cosine_angle = torch.clamp(dot_product / magnitudes, -1.0, 1.0)
    angle_rad = torch.acos(cosine_angle)
    angle_deg = torch.rad2deg(angle_rad)
    return angle_deg.item()

def init_weights(module):
    """Initializes weights like in NanoGPT."""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

def create_vectors(num_vectors, vector_dim):
    """Creates a matrix of vectors (num_vectors x vector_dim)."""
    dummy_layer = nn.Linear(vector_dim, num_vectors)
    init_weights(dummy_layer)
    return dummy_layer.weight.data

def measure_crowding(vectors):
    """Measures the crowding of vectors.  Returns a list of minimum angles."""
    num_vectors = vectors.size(0)
    min_angles = []
    selected_vectors = []

    for i in range(num_vectors):
        current_vector = vectors[i]
        selected_vectors.append(current_vector)

        if i > 0:
            min_angle = 360.0
            for j in range(i):
                angle = calculate_angle(current_vector, selected_vectors[j])
                min_angle = min(min_angle, angle)
            min_angles.append(min_angle)
    return min_angles

def optimize_angles(vectors, iterations=100, lr=0.01):
    """Optimizes the angles between vectors to be as close to 90 degrees as possible."""
    vectors = vectors.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([vectors], lr=lr)

    for iteration in range(iterations):
        optimizer.zero_grad()
        loss = 0

        for i in range(vectors.size(0)):
            for j in range(i + 1, vectors.size(0)):
                dot_product = torch.dot(vectors[i], vectors[j])
                magnitudes = torch.norm(vectors[i]) * torch.norm(vectors[j])

                if magnitudes > 0:
                    cosine_angle = torch.clamp(dot_product / magnitudes, -1.0, 1.0)
                    angle_rad = torch.acos(cosine_angle)
                    angle_deg = torch.rad2deg(angle_rad)

                    # Loss: Squared difference from 90 degrees
                    loss += (angle_deg - 90.0) ** 2

                else:
                    loss += torch.sum((vectors[i] - vectors[j])**2)


        loss.backward()
        optimizer.step()

        with torch.no_grad():
            vectors.data = vectors.data / torch.norm(vectors.data, dim=1, keepdim=True)

    return vectors.detach()



def main(args):
    sns.set_theme(style="whitegrid")
    output_dir = "output_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_original_min_angles = {}
    all_optimized_min_angles = {}

    for dim in args.dimensions:
        vectors = create_vectors(args.num_vectors, dim)
        original_min_angles = measure_crowding(vectors)
        all_original_min_angles[dim] = np.array(original_min_angles)

        if args.optimize:
            optimized_vectors = optimize_angles(vectors, iterations=args.iterations, lr=args.lr)
            optimized_min_angles = measure_crowding(optimized_vectors)
            all_optimized_min_angles[dim] = np.array(optimized_min_angles)

    np.save(os.path.join(output_dir, "original_min_angles.npy"), all_original_min_angles)
    if args.optimize:
        np.save(os.path.join(output_dir, "optimized_min_angles.npy"), all_optimized_min_angles)

    # --- Plotting (using loaded data) ---

    # First plot: Original vectors
    plt.figure(figsize=(12, 8))
    loaded_original_min_angles = np.load(os.path.join(output_dir, "original_min_angles.npy"), allow_pickle=True).item()
    for dim, min_angles in loaded_original_min_angles.items():
        sns.lineplot(x=range(1, len(min_angles) + 1), y=min_angles, label=f'Dim: {dim} (Original)', linewidth=2, linestyle="--")

    plt.xlabel("Number of Vectors Added", fontsize=14)
    plt.ylabel("Minimum Angle (Degrees)", fontsize=14)
    plt.title("Vector Crowding (Original)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(1, args.num_vectors + 1)
    plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "original_vectors.png"))
    plt.close()

    # Second plot: Original and Optimized (if requested)
    if args.optimize:
        plt.figure(figsize=(12, 8))
        loaded_optimized_min_angles = np.load(os.path.join(output_dir, "optimized_min_angles.npy"), allow_pickle=True).item()

        for dim in args.dimensions:
            if dim in loaded_original_min_angles:
                sns.lineplot(x=range(1, len(loaded_original_min_angles[dim]) + 1), y=loaded_original_min_angles[dim], label=f'Dim: {dim} (Original)', linewidth=2, linestyle="--")
            if dim in loaded_optimized_min_angles:
                sns.lineplot(x=range(1, len(loaded_optimized_min_angles[dim]) + 1), y=loaded_optimized_min_angles[dim], label=f'Dim: {dim} (Optimized)', linewidth=2)

        plt.xlabel("Number of Vectors Added", fontsize=14)
        plt.ylabel("Minimum Angle (Degrees)", fontsize=14)
        plt.title("Vector Crowding (Original vs. Optimized)", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(1, args.num_vectors + 1)
        plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, framealpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "optimized_vectors.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze vector crowding across different dimensions.")
    parser.add_argument('--dimensions', nargs='+', type=int, default=[2, 4, 16, 32, 64, 128, 256, 512],
                        help='List of vector dimensions to analyze.')
    parser.add_argument('--num_vectors', type=int, default=1000,
                        help='Number of vectors to analyze for each dimension.')
    parser.add_argument('--optimize', action='store_true',
                        help='Enable optimization of vector angles.')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of optimization iterations.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimization.')

    args = parser.parse_args()
    main(args)

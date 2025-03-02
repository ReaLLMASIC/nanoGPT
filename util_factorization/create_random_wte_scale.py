import torch
import numpy as np
import os
import argparse

def save_random_matrices(vocab_size, emb_size, A, mean, stddev, output_dir):
    """
    Initializes two random matrices W1 and W2 with given dimensions and saves them.
    W1: vocab_size x A
    W2: A x emb_size
    """
    W1 = torch.normal(mean=mean, std=stddev, size=(vocab_size, A)).cpu().numpy()
    W2 = torch.normal(mean=mean, std=stddev, size=(A, emb_size)).cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)

    # Save W1 and W2
    wte_file_path = os.path.join(output_dir, f"{A}_wte.npy")
    scale_matrices_file_path = os.path.join(output_dir, f"{A}_scale_matrices.npz")

    np.save(wte_file_path, W1)
    np.savez(scale_matrices_file_path, scale_up=W2, scale_down=W2)
    print(f"Saved W1 to {wte_file_path} and W2 to {scale_matrices_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Random matrix initialization script.")
    parser.add_argument('--A', type=int, default=128, help="Dimensionality of the hidden layer A (default: 128).")
    parser.add_argument('--mean', type=float, default=0.0, help="Mean of the random initialization (default: 0.0).")
    parser.add_argument('--stddev', type=float, default=0.02, help="Standard deviation of the random initialization (default: 0.02).")
    parser.add_argument('--vocab_size', type=int, default=50257, help="Vocabulary size (default: 50257).")
    parser.add_argument('--emb_size', type=int, default=768, help="Embedding size (default: 768).")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save the random matrices.")

    args = parser.parse_args()

    # Create random matrices W1 (vocab_size x A) and W2 (A x emb_size)
    save_random_matrices(args.vocab_size, args.emb_size, args.A, args.mean, args.stddev, args.output_dir)

if __name__ == "__main__":
    main()


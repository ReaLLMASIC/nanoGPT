#!/usr/bin/env python3

import argparse
import math
import torch
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate hypercube corners for embeddings in PyTorch.")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Select mode: direct or corners")

    # Mode 1: Direct dimension
    parser_direct = subparsers.add_parser("direct", help="Specify dimension directly.")
    parser_direct.add_argument(
        "--dim",
        type=int,
        required=True,
        help="Dimension of hypercube."
    )
    parser_direct.add_argument(
        "--max_vectors",
        type=int,
        default=None,
        help="max number of vectors to generate (optional in direct mode)"
    )

    # Mode 2: From number of vectors
    parser_vectors = subparsers.add_parser("corners", help="Specify number of vectors, compute dimension from it.")
    parser_vectors.add_argument(
        "--num_vectors",
        type=int,
        required=True,
        help="Number of vectors (minimum) to fit in a hypercube's corners."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="corners.npy",
        help="Path to save the generated corners tensor (Numpy format)."
    )

    args = parser.parse_args()

    if args.mode == "direct":
        dim = args.dim
        n_corners = 2 ** dim
        if args.max_vectors is not None:
            n_corners = min(n_corners, args.max_vectors)
        print(f"Mode: direct - dimension={dim}, total_corners={n_corners}")
    elif args.mode == "corners":
        num_vectors = args.num_vectors
        # Calculate smallest dimension d such that 2^d >= num_vectors
        dim = math.ceil(math.log2(num_vectors))
        n_corners = 2 ** dim
        print(f"Mode: corners - requested={num_vectors}, dimension={dim}, total_corners={n_corners}")

    # Create the corner vectors as a FloatTensor of shape [n_corners, dim].
    # Each corner is a binary vector with 0/1 entries.
    corners = torch.zeros((n_corners, dim), dtype=torch.float32)

    for i in tqdm(range(n_corners), desc="Generating corners"):
        for d in range(dim):
            # (i >> d) & 1 extracts the d-th bit of i
            corners[i, d] = (i >> d) & 1

    # Optionally, if you only need exactly `num_vectors` corners in the corners mode:
    # corners = corners[:num_vectors]
    print(corners)

    # Save to disk
    np.save(args.output_path, corners.numpy())
    print(f"Saved corners to {args.output_path}")

if __name__ == "__main__":
    main()


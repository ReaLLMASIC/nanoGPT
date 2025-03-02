import numpy as np
import argparse

def view_keys(npz_file):
    # Load the .npz file
    data = np.load(npz_file)

    # List all keys in the .npz file
    keys = data.files

    print(f"Keys in {npz_file}:")
    for key in keys:
        print(f"- {key}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="View keys in an .npz file")
    parser.add_argument("npz_file", type=str, help="Path to the .npz file")

    # Parse arguments
    args = parser.parse_args()

    # View keys in the specified npz file
    view_keys(args.npz_file)


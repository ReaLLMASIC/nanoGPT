import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def load_npy_file(file_path):
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def display_heatmap(data, cmap='viridis', save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap=cmap, annot=False, fmt="g")

    plt.title("Numpy Array Heatmap")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")

    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="View a .npy file as a heatmap using Seaborn.")
    parser.add_argument("file_path", type=str, help="Path to the .npy file.")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap to use for the heatmap (default: viridis).")
    parser.add_argument("--save", type=str, default=None, help="Path to save the heatmap image (optional).")

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"File not found: {args.file_path}")
        return

    data = load_npy_file(args.file_path)

    if data is None:
        return

    if data.ndim != 2:
        print(f"Unsupported data shape: {data.shape}. Only 2D arrays can be visualized as a heatmap.")
        return

    display_heatmap(data, cmap=args.cmap, save_path=args.save)

if __name__ == "__main__":
    main()


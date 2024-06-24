import numpy as np
import argparse
from tabulate import tabulate

def export_npy_to_text(npy_path, output_path, digits):
    # Load the .npy file
    data = np.load(npy_path)
    
    with open(output_path, 'w') as f:
        if data.ndim == 1:
            table = [[idx, f"{value:.{digits}f}"] for idx, value in enumerate(data)]
            f.write(tabulate(table, headers=["Index", "Value"], tablefmt="grid"))
        elif data.ndim == 2:
            table = [[row_idx] + [f"{value:.{digits}f}" for value in row] for row_idx, row in enumerate(data)]
            headers = ["Row"] + [f"Col {i}" for i in range(data.shape[1])]
            f.write(tabulate(table, headers=headers, tablefmt="grid"))
        elif data.ndim == 3:
            table = [
                [depth_idx, row_idx] + [f"{value:.{digits}f}" for value in row]
                for depth_idx, depth in enumerate(data)
                for row_idx, row in enumerate(depth)
            ]
            headers = ["Depth", "Row"] + [f"Col {i}" for i in range(data.shape[2])]
            f.write(tabulate(table, headers=headers, tablefmt="grid"))
        else:
            f.write("Data has more than 3 dimensions, which is not supported.\n")

def main():
    parser = argparse.ArgumentParser(description="Export .npy file contents to a formatted text file.")
    parser.add_argument('--npy_path', type=str, required=True, help="Path to the .npy file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the output text file.")
    parser.add_argument('--digits', type=int, default=4, help="Number of digits to display for values.")
    args = parser.parse_args()
    
    export_npy_to_text(args.npy_path, args.output_path, args.digits)

if __name__ == "__main__":
    main()


import numpy as np
import argparse

def export_npy_to_text(npy_path, output_path, digits):
    # Load the .npy file
    data = np.load(npy_path)
    
    with open(output_path, 'w') as f:
        if data.ndim == 1:
            f.write(f"{'Index':>10} {'Value':>10}\n")
            f.write("="*22 + "\n")
            for idx, value in enumerate(data):
                f.write(f"{idx:>10} {value:.{digits}f}\n")
        elif data.ndim == 2:
            f.write(f"{'Row':>5} {'Col':>5} {'Value':>10}\n")
            f.write("="*22 + "\n")
            for row_idx, row in enumerate(data):
                for col_idx, value in enumerate(row):
                    f.write(f"{row_idx:>5} {col_idx:>5} {value:.{digits}f}\n")
        elif data.ndim == 3:
            f.write(f"{'Depth':>5} {'Row':>5} {'Col':>5} {'Value':>10}\n")
            f.write("="*32 + "\n")
            for depth_idx, depth in enumerate(data):
                for row_idx, row in enumerate(depth):
                    for col_idx, value in enumerate(row):
                        f.write(f"{depth_idx:>5} {row_idx:>5} {col_idx:>5} {value:.{digits}f}\n")
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


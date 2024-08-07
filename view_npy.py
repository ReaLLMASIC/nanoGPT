import numpy as np
from rich.console import Console
from rich.table import Table
import argparse

def view_npy(npy_path, digits):
    # Load the .npy file
    data = np.load(npy_path)

    # Initialize the console
    console = Console()

    # Create a table
    table = Table(title=f"Contents of {npy_path}")

    # Determine the number of dimensions and add corresponding columns
    if data.ndim == 1:
        table.add_column("Index", justify="right", style="cyan")
        table.add_column("Value", justify="right", style="magenta")

        # Add rows to the table
        for idx, value in enumerate(data):
            table.add_row(
                str(idx),
                f"{value:.{digits}f}"
            )
    elif data.ndim == 2:
        table.add_column("Row", justify="right", style="cyan")
        for col_idx in range(data.shape[1]):
            table.add_column(f"Col {col_idx}", justify="right", style="magenta")

        # Add rows to the table
        for row_idx, row in enumerate(data):
            table.add_row(
                str(row_idx),
                *[f"{value:.{digits}f}" for value in row]
            )
    elif data.ndim == 3:
        table.add_column("Depth", justify="right", style="cyan")
        table.add_column("Row", justify="right", style="cyan")
        for col_idx in range(data.shape[2]):
            table.add_column(f"Col {col_idx}", justify="right", style="magenta")

        # Add rows to the table
        for depth_idx, depth in enumerate(data):
            for row_idx, row in enumerate(depth):
                table.add_row(
                    str(depth_idx),
                    str(row_idx),
                    *[f"{value:.{digits}f}" for value in row]
                )
    else:
        console.print("Data has more than 3 dimensions, which is not supported.", style="bold red")
        return

    # Print the table
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="View .npy file contents with rich formatting.")
    parser.add_argument('--npy_path', type=str, required=True, help="Path to the .npy file.")
    parser.add_argument('--digits', type=int, default=4, help="Number of digits to display for values.")
    args = parser.parse_args()

    view_npy(args.npy_path, args.digits)

if __name__ == "__main__":
    main()


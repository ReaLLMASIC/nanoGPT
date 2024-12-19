# Matrix Factorization with Vizier Optimization

This script performs matrix factorization using PyTorch with a configurable rank `A` and optimizes the factorization process using Google's Vizier service. The optimization is carried out over multiple iterations, where each iteration tries different values of `A` to minimize the reconstruction loss. The results are logged and saved to a CSV file for further analysis.

## Features

- **Matrix Factorization**: The script factorizes an input matrix into two lower-rank matrices using gradient descent.
- **Vizier Optimization**: Utilizes Google's Vizier service to find the optimal rank `A` that minimizes the reconstruction loss.
- **Multiple Seeds**: Runs the factorization multiple times with different random seeds to ensure robustness.
- **Progress Visualization**: Real-time progress bars and loss updates using the Rich library.
- **CSV Output**: Saves the results after each iteration to a CSV file.

## Requirements

- Python 3.8 or higher
- PyTorch
- NumPy
- Pandas
- Rich
- Google's Vizier Service (`vizier.service` package)

You can install the required Python packages using pip:

```bash
pip install torch numpy pandas rich vizier
```

## Usage

### Command-Line Arguments

The script accepts several command-line arguments that allow you to customize the optimization and factorization process:

- `--vizier_algorithm`: The Vizier algorithm to use for optimization (default: `"GRID_SEARCH"`).
- `--vizier_iterations`: Number of Vizier iterations to perform (default: `20`).
- `--num_epochs`: Number of epochs for training the factorization model (default: `1000`).
- `--num_seeds`: Number of random seeds to try for each value of `A` (default: `5`).
- `--A_start`: Minimum value of `A` for optimization (default: `10`).
- `--A_end`: Maximum value of `A` for optimization (default: `100`).
- `--output_csv`: Path to the output CSV file where results will be saved (default: `"results.csv"`).
- `--matrix_path`: Path to a `.npy` file containing the matrix to be factorized. If not provided, a random matrix of shape `(50000, 384)` will be used.

### Example Usage

```bash
python factorization_script.py --vizier_algorithm GRID_SEARCH --vizier_iterations 10 --num_epochs 500 --num_seeds 3 --A_start 20 --A_end 50 --output_csv results.csv --matrix_path /path/to/matrix.npy
```

This example command will:

- Use the `GRID_SEARCH` algorithm for Vizier optimization.
- Run 10 Vizier iterations, with 500 epochs each.
- Test values of `A` between 20 and 50.
- Use 3 different random seeds for each value of `A`.
- Save the results to `results.csv`.
- Factorize the matrix stored at `/path/to/matrix.npy`.

### Output

- **Results Table**: After each iteration, a table showing the minimum, maximum, mean, median, and standard deviation of the losses for each value of `A` is displayed.
- **CSV File**: The results of all Vizier iterations are saved to the specified CSV file, including statistics like the minimum, maximum, mean, median, and standard deviation of the loss for each value of `A`.

### Example Output CSV

| A  | Min Loss | Max Loss | Mean Loss | Median Loss | Std Dev |
|----|----------|----------|-----------|-------------|---------|
| 20 | 0.1234   | 0.2345   | 0.1789    | 0.1765      | 0.0300  |
| 30 | 0.0987   | 0.1456   | 0.1201    | 0.1198      | 0.0150  |
| ...| ...      | ...      | ...       | ...         | ...     |

### Notes

- If the matrix path is not provided, the script will generate a random matrix with a shape of `(50000, 384)` for factorization.
- Ensure that your environment is set up to access Vizier services.

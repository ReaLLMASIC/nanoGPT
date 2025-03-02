import argparse
import csv
import numpy as np
from typing import List

def map_range_to_01(start: float, end: float, steps: int) -> List[float]:
    return list(np.linspace(start, end, steps))

def add_padding(value: float, padding_count: int) -> List[str]:
    return [str(value)] + ['r'] * padding_count

def generate_csv(mapped_values: List[float], padding_count: int, output_file: str) -> None:
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for value in mapped_values:
            padded_row = add_padding(value, padding_count)
            writer.writerow(padded_row)
    print(f"CSV file '{output_file}' generated with {len(mapped_values)} rows, each having dimension {padding_count + 1}.")

def main():
    parser = argparse.ArgumentParser(description='Generate a CSV file with each number on its own row, padded with "r" values.')
    parser.add_argument('--start', type=float, required=True, help='Starting value of the range.')
    parser.add_argument('--end', type=float, required=True, help='Ending value of the range.')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps in the range.')
    parser.add_argument('--padding_count', type=int, required=True, help='Number of "r" paddings to add to each row.')
    parser.add_argument('--output', type=str, default='output.csv', help='Output CSV file name (default: output.csv)')
    args = parser.parse_args()

    # Map the range to 0.0 - 1.0
    mapped_values = map_range_to_01(args.start, args.end, args.steps)

    # Generate the CSV file with padding
    generate_csv(mapped_values, args.padding_count, args.output)

if __name__ == "__main__":
    main()


import numpy as np
import random
import argparse
import csv
from typing import List, Tuple, Dict
from rich import print

def generate_letter_mapping(degrees: int, letter_offset: float) -> Dict[str, Tuple[float, float]]:
    radians = np.deg2rad(degrees)
    offset_radians = np.deg2rad(letter_offset)
    cos, sin = np.cos(radians + offset_radians), np.sin(radians + offset_radians)
    cos_centered, sin_centered = np.cos(offset_radians), np.sin(offset_radians)
    return {
        'H': (cos, sin),
        'M': (cos_centered, sin_centered),
        'L': (cos, -sin),
        'y': (cos, sin),
        'n': (cos, -sin),
        's': (cos, sin),
        'a': (cos_centered, sin_centered),
        'f': (cos, -sin),
        '<': (-1, 0),
    }

def random_value(mean: float = 0.0, stdev: float = 0.02) -> float:
    return np.random.normal(mean, stdev)

def random_value_pair(mean: float = 0.0, stdev: float = 0.02) -> Tuple[float, float]:
    return np.random.normal(mean, stdev), np.random.normal(mean, stdev)

def map_numeric_to_circle(value: float, max_angle_difference: float, number_offset: float) -> Tuple[float, float]:
    max_radian_difference = np.deg2rad(max_angle_difference)
    half_max_radian_difference = max_radian_difference / 2.0
    centered_angle = max_radian_difference * value - half_max_radian_difference
    final_angle = centered_angle + np.deg2rad(number_offset)
    return np.cos(final_angle), np.sin(final_angle)

def map_random_to_circle(mean: float, stdev: float, random_offset: float) -> Tuple[float, float]:
    radians = random_value(mean, stdev)
    radians += np.deg2rad(random_offset)
    return np.cos(radians), np.sin(radians)

def map_value(value: str, letter_mapping: Dict[str, Tuple[float, float]],
              max_angle_difference: float, letter_offset: float, number_offset:
              float, random_offset: float, mean: float, stdev: float,
              random_value_pair_flag: bool, map_random_to_unit_circle_flag:
              bool, direct_num_mapping: bool) -> List[float]:
    if value.lower() == 'r':
        if map_random_to_unit_circle_flag:
            return list(map_random_to_circle(mean, stdev, random_offset))
        elif random_value_pair_flag:
            return list(random_value_pair(mean, stdev))
        else:
            return [random_value(mean, stdev)]
    try:
        numeric_value = float(value)
        if direct_num_mapping:
            print(numeric_value)
            return [numeric_value]
        else:
            if 0 <= numeric_value <= 1:
                print("Warning: out of range outputs detected")
            return list(map_numeric_to_circle(numeric_value, max_angle_difference, number_offset))
    except ValueError:
        return list(letter_mapping.get(value, (random_value(mean, stdev), random_value(mean, stdev))))

def load_csv(file_path: str) -> List[List[str]]:
    with open(file_path, newline='') as csvfile:
        return list(csv.reader(csvfile))

def map_table(table: List[List[str]], letter_mapping: Dict[str, Tuple[float, float]], max_angle_difference: float, letter_offset: float, number_offset: float, random_offset: float, mean: float, stdev: float, random_value_pair_flag: bool, map_random_to_unit_circle_flag: bool, direct_num_mapping: bool) -> np.ndarray:
    return np.array([sum([map_value(value, letter_mapping, max_angle_difference, letter_offset, number_offset, random_offset, mean, stdev, random_value_pair_flag, map_random_to_unit_circle_flag, direct_num_mapping) for value in row], []) for row in table])

def main():
    parser = argparse.ArgumentParser(description='Generate initial_wte.npy from a CSV file.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--direct_num_mapping', default=False, action=argparse.BooleanOptionalAction, help='Keep numerical values as original values, may need to be used in conjunction with no-random_value_pair for correct output dimensions.')
    parser.add_argument('--degrees', type=int, default=60, help='Degrees of separation for letters (default: 60)')
    parser.add_argument('--letter_offset', type=float, default=0.0, help='Offset angle for the letter mapping (default: 0.0)')
    parser.add_argument('--number_offset', type=float, default=0.0, help='Offset angle for numeric mapping (default: 0.0)')
    parser.add_argument('--random_offset', type=float, default=0.0, help='Offset angle for random value mapping (default: 0.0)')
    parser.add_argument('--random_value_pair', default=True, action=argparse.BooleanOptionalAction, help="Use two random values per 'r' input (default: True)")
    parser.add_argument('--map_random_to_unit_circle', default=False, action=argparse.BooleanOptionalAction, help="Map 'r' input to unit circle")
    parser.add_argument('--mean', type=float, default=0.0, help='Mean for random number generation (default: 0.0)')
    parser.add_argument('--stdev', type=float, default=0.02, help='Standard deviation for random number generation (default: 0.02)')
    parser.add_argument('--max_angle_difference', type=float, default=180.0, help='Maximum angle difference for numeric mapping (default: 180.0)')
    parser.add_argument('--output', type=str, default='initial_wte.npy', help='Output file name (default: initial_wte.npy)')
    args = parser.parse_args()

    letter_mapping = generate_letter_mapping(args.degrees, args.letter_offset)
    table = load_csv(args.csv)

    wte = map_table(table, letter_mapping, args.max_angle_difference,
                    args.letter_offset, args.number_offset, args.random_offset,
                    args.mean, args.stdev, args.random_value_pair,
                    args.map_random_to_unit_circle, args.direct_num_mapping)

    print(f"Shape of wte: {wte.shape}")
    np.save(args.output, wte)
    print(f"Saved initial wte with shape {wte.shape} to {args.output}")

    np.set_printoptions(precision=3, suppress=True)
    print("\nPrint wte (3 decimal places):")
    print(wte)

if __name__ == "__main__":
    main()


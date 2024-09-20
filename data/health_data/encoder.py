import argparse
import random
import re

# Define prefixes for each field
prefixes = ['y', 'd', 'w', 'H', 'M', 'S', 'b', 'm', 'p', 'o']

# Create a map to associate prefixes with indices
prefix_to_index = {prefix: i for i, prefix in enumerate(prefixes)}

def add_prefix_and_shuffle(data, shuffle_columns=None, reverse=False):
    new_data = []

    for row in data:
        # Split each row by commas and add prefixes
        values = row.strip().split(',')
        prefixed_row = [f"{prefix}{val}" for prefix, val in zip(prefixes, values)]

        if reverse:
            # Systematic reversal: Extract prefix, sort by prefix's position in original order
            sorted_row = sorted(prefixed_row, key=lambda x: prefix_to_index[x[0]])
            new_data.append(''.join(sorted_row))  # Join without commas
        else:
            if shuffle_columns:
                # Shuffle only selected columns based on user input
                indices_to_shuffle = [prefix_to_index[col] for col in shuffle_columns if col in prefix_to_index]
                to_shuffle = [prefixed_row[i] for i in indices_to_shuffle]
                random.shuffle(to_shuffle)
                # Replace the shuffled elements back in the original list
                for i, idx in enumerate(indices_to_shuffle):
                    prefixed_row[idx] = to_shuffle[i]
            # Add the processed row
            new_data.append(''.join(prefixed_row))  # Join without commas

    return new_data

def parse_prefixed_row(prefixed_row):
    """
    Parses a row with prefixed data and returns it in the correct original order.
    Assumes that the row is prefixed and unshuffled.
    """
    # Use a regex to extract the prefix (first character) and value
    parsed_values = {}
    for value in re.findall(r'([a-zA-Z])(\d+\.?\d*)', prefixed_row):
        prefix, num = value
        parsed_values[prefix_to_index[prefix]] = num

    # Rebuild the row in the original order based on index
    return ','.join([parsed_values[i] for i in sorted(parsed_values.keys())])

def reverse_operation(data):
    """
    Reverses the operation by recognizing prefixed values
    and restoring the original order.
    """
    reversed_data = []
    for row in data:
        reversed_row = parse_prefixed_row(row)
        reversed_data.append(reversed_row)

    return reversed_data

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process and shuffle prefixed data.")
    parser.add_argument('input_file', help="Input file containing the CSV data.")
    parser.add_argument('output_file', help="Output file to save the processed data.")
    parser.add_argument('--shuffle-columns', nargs='+', help="Specify columns to shuffle (e.g., b m p o for bpm, movement, pi, spo2). Default is no shuffling of time columns.")
    parser.add_argument('--reverse', action='store_true', help="Reverse the operation to restore original order.")

    args = parser.parse_args()

    # Read the input file
    with open(args.input_file, 'r') as f:
        data = f.readlines()

    if args.reverse:
        # Reverse operation: restore the original order
        processed_data = reverse_operation(data)
    else:
        # Regular operation: shuffle and prefix the data
        processed_data = add_prefix_and_shuffle(data, shuffle_columns=args.shuffle_columns)

    # Write the output to the file
    with open(args.output_file, 'w') as f:
        for row in processed_data:
            f.write(row + '\n')

if __name__ == "__main__":
    main()


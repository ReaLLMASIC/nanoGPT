import numpy as np
import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description='Load and print a NumPy array from a .npy file')
parser.add_argument('npy_file', type=str, help='Path to the .npy file')

# Parse the arguments
args = parser.parse_args()

# Load the .npy file
array = np.load(args.npy_file)

# Print the NumPy array
print("Loaded array:")
print(array)
print(array.shape)


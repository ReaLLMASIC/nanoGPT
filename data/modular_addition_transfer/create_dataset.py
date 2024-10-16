import argparse
import random

def create_digit_map(offset, num_digits):
    """
    Creates a dictionary that maps digits (0-num_digits) to characters with a specified UTF offset.
    Example: offset = 97 for 'a' -> 'j' mapping (if num_digits = 10).
    """
    digit_map = {}
    for i in range(num_digits):
        digit_map[i] = chr(offset + i)
    return digit_map

def modular_addition(a, b, mod):
    """
    Performs modular addition of two numbers.
    """
    return (a + b) % mod

def represent_number(num, digit_map):
    """
    Represents the number using the digit map provided.
    Example: 123 -> 'abc' if map maps 1->'a', 2->'b', 3->'c'.
    """
    return ''.join(digit_map[int(digit)] for digit in str(num))

def all_modular_additions(mod, offset, num_digits, shuffle_output=True):
    """
    Generates all modular additions for two numbers 0-(num_digits-1), 
    and allows for different representations by changing the UTF-8 offset.
    
    Saves the result into a text file named based on the UTF-8 offset. Optionally shuffles the result.
    """
    digit_map = create_digit_map(offset, num_digits)
    file_name = f"input_{offset}.txt"
    
    results = []
    
    for i in range(mod):
        for j in range(mod):
            result = modular_addition(i, j, mod)
            i_rep = represent_number(i, digit_map)
            j_rep = represent_number(j, digit_map)
            result_rep = represent_number(result, digit_map)
            
            # Format the result without spaces or (mod _) part
            results.append(f"{i_rep}+{j_rep}={result_rep}\n")
    
    # Shuffle the results if shuffle_output is True
    if shuffle_output:
        random_seed = offset  # Use the offset as a unique random seed for each file
        random.seed(random_seed)
        random.shuffle(results)

    # Write the results to the file
    with open(file_name, "w") as f:
        f.writelines(results)

    print(f"Saved results for offset {offset} in {file_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate modular additions with customizable representations.")
    parser.add_argument("-u", "--unoverlapped-offsets", type=int, default=1, help="Number of unoverlapped offsets.")
    parser.add_argument("-m", "--max-digits", type=int, default=10, help="Max number of digits for modular arithmetic.")
    parser.add_argument("-o", "--start-offset", type=int, default=48, help="Starting UTF offset (default 48 for '0').")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the output file before writing (default True).")

    args = parser.parse_args()

    unoverlapped_offsets = args.unoverlapped_offsets
    max_digits = args.max_digits
    start_offset = args.start_offset
    shuffle_output = args.shuffle

    # Generate and save the modular additions for each offset
    for rep in range(unoverlapped_offsets):
        current_offset = start_offset + (rep * 10)
        all_modular_additions(max_digits, current_offset, max_digits, shuffle_output=shuffle_output)

if __name__ == "__main__":
    main()


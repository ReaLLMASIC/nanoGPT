import string
import argparse
import re
import nltk

# Make sure to install NLTK if using the 'pos' method and download taggers:
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# ------------------------------------------------------------------------
# Original "basic" transformation function
def transform_basic(text):
    punctuation = string.punctuation
    vowels = 'aeiouAEIOU'
    consonants = ''.join([c for c in string.ascii_letters if c not in vowels])

    transformed = []
    for char in text:
        if char in punctuation:
            transformed.append('1')
        elif char in vowels:
            transformed.append('2')
        elif char in consonants:
            transformed.append('3')
        else:
            transformed.append('_')
    return ''.join(transformed)

# ------------------------------------------------------------------------
# "pos" transformation function (optional, requires nltk)
def transform_pos(text):
    # Split so that whitespace is preserved as separate chunks
    chunks = re.split(r'(\s+)', text)
    transformed = []

    for chunk in chunks:
        if not chunk:
            continue
        if chunk.isspace():
            # Turn each whitespace character into '_'
            transformed.append('_' * len(chunk))
        else:
            # Tag this chunk (treated as one token)
            tagged = nltk.pos_tag([chunk])[0]  # (token, POS)
            _, pos = tagged
            # Use the first letter of the POS tag, in lowercase, repeated per character
            letter_for_pos = pos[0].lower()
            transformed.append(letter_for_pos * len(chunk))

    return ''.join(transformed)

# ------------------------------------------------------------------------
# New "position" transformation with single-character-per-position
def transform_position(text):
    """
    For each non-whitespace 'word', replace each character with
    a single character representing its 1-based position:
      Positions 1-9  -> '1'..'9'
      Positions 10-35 -> 'A'..'Z'
      Positions 36-61 -> 'a'..'z'
      If you exceed 61, wrap around to the start (modulo).
    Whitespace remains underscores.
    Example:
      " dog cat" -> "_123_123"
    """

    # Our lookup table for positions (you can adjust or expand as desired)
    #  - 1..9 = digits
    #  - 10..35 = uppercase letters
    #  - 36..61 = lowercase letters
    position_chars = (
        "123456789"            # (pos 1-9)
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # (pos 10-35)
        "abcdefghijklmnopqrstuvwxyz"  # (pos 36-61)
    )
    max_index = len(position_chars)  # 61

    # Split so that whitespace is preserved as separate chunks
    chunks = re.split(r'(\s+)', text)
    transformed = []

    for chunk in chunks:
        if not chunk:
            # Empty string, possibly from a split edge
            continue

        if chunk.isspace():
            # Turn each whitespace character into '_'
            transformed.append('_' * len(chunk))
        else:
            # Replace each character by the appropriate single char from our table
            out = []
            for i, _ in enumerate(chunk, start=1):
                # i is 1-based position, modulo with the table length
                # (i - 1) % max_index ensures we start from index 0
                idx = (i - 1) % max_index
                out.append(position_chars[idx])
            transformed.append(''.join(out))

    return ''.join(transformed)

# ------------------------------------------------------------------------
def transform_file(filename, method):
    """
    Transforms a file in-place using the selected method.
    """
    try:
        with open(filename, 'r+', encoding='utf-8') as file:
            # Read the entire file content
            file_content = file.read()

            if method == 'basic':
                transformed_content = transform_basic(file_content)
            elif method == 'pos':
                transformed_content = transform_pos(file_content)
            elif method == 'position':
                transformed_content = transform_position(file_content)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Overwrite file with transformed text
            file.seek(0)
            file.write(transformed_content)
            file.truncate()

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")

# ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform a text file by replacing characters.")
    parser.add_argument("input_file", help="The input text file to transform.")
    parser.add_argument(
        "--method", 
        choices=["basic", "pos", "position"],
        default="basic",
        help="Which transformation method to use."
    )
    args = parser.parse_args()

    transform_file(args.input_file, args.method)


import string
import argparse
import re

# For the 'pos' method:
# Make sure to install NLTK (pip install nltk) and download the "averaged_perceptron_tagger" resource:
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
import nltk

def transform_basic(text):
    """
    The original basic transformation:
      - Punctuation -> '1'
      - Vowels -> '2'
      - Consonants -> '3'
      - Everything else -> '_'
    """
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
            # Whitespace, digits, etc.
            transformed.append('_')
    return ''.join(transformed)


def transform_pos(text):
    """
    Naive POS-based transformation:
      - Split the text into 'chunks' (words vs. whitespace) using a regex that preserves delimiters.
      - For each non-whitespace chunk, run nltk.pos_tag([chunk]) to get a single (token, tag) pair.
      - Convert the tag's first letter to lowercase (e.g. 'NN' -> 'n', 'VB' -> 'v') and
        repeat it for each character in the chunk.
      - For whitespace chunks, output underscores.
      - Example: " dog cat" -> "_nnn_nnn"
    """
    # Split so that whitespace is kept as separate chunks
    chunks = re.split(r'(\s+)', text)
    transformed = []

    for chunk in chunks:
        if not chunk:
            # Empty string (beginning or end) – skip or turn into ''
            continue
        if chunk.isspace():
            # Turn each whitespace character into '_'
            transformed.append('_' * len(chunk))
        else:
            # Tag this chunk as a single token (naive approach)
            # If the chunk includes punctuation, pos_tag might label it differently,
            # but we'll keep it simple here.
            tagged = nltk.pos_tag([chunk])[0]  # returns (word, POS)
            word, pos = tagged
            # Take the first letter of the POS tag in lowercase
            letter_for_pos = pos[0].lower()
            # Repeat for each character in the chunk
            transformed.append(letter_for_pos * len(chunk))

    return ''.join(transformed)


def transform_position(text):
    """
    Positional transformation:
      - For each 'word' chunk (non-whitespace), replace each character with its 1-based position in that chunk.
      - Whitespace becomes underscores.
      - Example: " dog cat" -> "_123_123"
    """
    chunks = re.split(r'(\s+)', text)
    transformed = []

    for chunk in chunks:
        if not chunk:
            continue
        if chunk.isspace():
            # Replace each space with an underscore
            transformed.append('_' * len(chunk))
        else:
            # Replace each character by 1-based position
            out = []
            for i, _ in enumerate(chunk):
                out.append(str(i+1))
            transformed.append(''.join(out))

    return ''.join(transformed)


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


import argparse
import os
import json
from tqdm import tqdm

def append_to_file(file_path, field, snac):
    """Append text data to file incrementally."""
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write('#B:' + snac )
            file.write('#U:' + field + '\n')
    else:
        with open(file_path, 'w') as file:
            file.write('#B:' + snac)
            file.write('#U:' + field + '\n')


def process_text_extraction(input_path, extract_text, value, output_path, text_seperator='-'):
    """Process the text file specified and extract text and write it to a file."""
    with open(input_path, 'r') as file:
        data = json.load(file)

    for entry in data:
        field = entry[value]
        raw_snac = entry["sequential_snac_tokens"]
        snac = ""
        if extract_text:
            snac = '-'.join(str(num) for num in raw_snac)
        else:
            for i in raw_snac:
                num = f"<snac>{i}</snac>"
                snac = snac + num
        
        append_to_file(output_path, field, snac)


def process_directory(input_path, extract_text, value, output_path, text_seperator):
    """Process specified directory."""
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
    for filename in tqdm(json_files, desc="Processing json files"):
        file_path = os.path.join(input_path, filename)
        process_text_extraction(file_path, extract_text, value, output_path, text_seperator)


def main():
    parser = argparse.ArgumentParser(description="Extracting text data from snac_json output json file.")
    parser.add_argument("input", help="Input json file path to extract text data from.")
    parser.add_argument("-t", "--extract_text", action="store_true",
                        help="the extracted value is related to text when specifying")
    parser.add_argument("value", type=str, help="Field to extract value from.")
    parser.add_argument("output", help="Output path to store the result file.")
    parser.add_argument('--text_seperator', type=str, default='-', help='The key to seperate snac tokens for tiktoken (default: "-").')
    parser.add_argument('--directory', action='store_true', help="Process all json file in the input directory.")
    args = parser.parse_args()

    if args.directory:
        process_directory(args.input, args.extract_text, args.value, args.output, args.text_seperator)
    else:
        process_text_extraction(args.input, args.extract_text, args.value, args.output, args.text_seperator)

if __name__ == "__main__":
    main()

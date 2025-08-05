import json
import argparse

def extract_values_by_key(json_file, key, output_file):
    """
    Filter all values associated with a specific key from a JSON file
    for which its length + length of snac are greater than 1024
    and writes them to an output json file, with the same format.

    Args:
        json_file: Path to the input JSON file.
        key: The key to search for in the JSON data.
        output_file: Path to the output json file.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)

        output_data = []
        total = 0
        remaining = 0
        for entry in data:
            total += 1
            ipa = entry[key]
            ipa_length = len(ipa)
            if ipa_length + int(entry["snac_token_len"]) <= 1024:
                remaining += 1
                output_data.append(entry)

        # Write to the ouput file
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(output_data, f_out, ensure_ascii=False, indent=4)

        print(f"Input file filtered and saved to '{output_file}")
        percentage = (remaining / total) * 100
        print(f"Remaining percentage: {percentage}%")

    except FileNotFoundError:
        print(f"Error: Input file '{json_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract values of a specific key from a JSON file to a text file.")
    parser.add_argument("json_file", help="Path to the input JSON file.")
    parser.add_argument("key", help="The key whose values you want to contain.")
    parser.add_argument("output_file", help="Path to the output text file.")

    args = parser.parse_args()

    extract_values_by_key(args.json_file, args.key, args.output_file)
    print(f"Values for key '{args.key}' extracted to '{args.output_file}'")

if __name__ == "__main__":
    main()

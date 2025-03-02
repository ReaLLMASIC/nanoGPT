import argparse
import json
import re
from transformers import AutoTokenizer

def extract_all_tokens(tokenizer):
    vocab_size = tokenizer.vocab_size
    all_tokens = []
    all_token_ids = []

    for token_id in range(vocab_size):
        token = tokenizer.decode([token_id])
        all_tokens.append(token)
        all_token_ids.append(token_id)

    return all_tokens, all_token_ids

def categorize_tokens(tokens, token_ids, output_file, cjk_categorize):
    categories = {
        "chinese": [],
        "english": [],
        "korean": [],
        "japanese": [],
        "misc": []
    }

    all_vocab = {
            "vocab":[]
            }

    cjk_ranges = [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
        (0x2CEB0, 0x2EBEF), # CJK Unified Ideographs Extension F
        (0x30000, 0x3134F),  # CJK Unified Ideographs Extension G
        (0x31350, 0x323AF),   # CJK Unified Ideographs Extension H
        (0x2EBF0, 0x2EE5F),  # CJK Unified Ideographs Extension I
    ]

    for token, token_id in zip(tokens, token_ids):
        if cjk_categorize:
            try:
                if any(0x3040 <= ord(c) <= 0x30FF for c in token):
                    categories["japanese"].append({"token": token, "id": token_id})
                elif any(0xAC00 <= ord(c) <= 0xD7A3 for c in token):
                    categories["korean"].append({"token": token, "id": token_id})
                elif any(0x4E00 <= ord(c) <= 0x9FFF for c in token):
                    categories["chinese"].append({"token": token, "id": token_id})
                elif re.search(r'[a-zA-Z]', token):
                    categories["english"].append({"token": token, "id": token_id})
                else:
                    is_misc = True
                    for start, end in cjk_ranges:
                        if any(start <= ord(c) <= end for c in token):
                            categories["chinese"].append({"token": token, "id": token_id})
                            is_misc = False
                            break

                    if is_misc:
                        categories["misc"].append({"token": token, "id": token_id})

            except ValueError:
                print(f"Skipping invalid token (ValueError): {token}")
            except IndexError as e:
                print(f"Skipping invalid token (IndexError {e}): {token}")

        else:
            all_vocab["vocab"].append({"token": token, "id": token_id})


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(categories if cjk_categorize else all_vocab, f, indent=4, ensure_ascii=False)

    print(f"Categorized tokens written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Tokenizer to JSON Categorizer')
    parser.add_argument('--model', type=str, default="google/gemma-7b", help='Tokenizer model to use')
    parser.add_argument("--cjk_categorize", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('-o', '--output', type=str, default='categorized_tokens.json', help='Path to the output JSON file.')

    args = parser.parse_args()

    print(f"Obtaining tokens from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    all_tokens, all_token_ids = extract_all_tokens(tokenizer)
    categorize_tokens(all_tokens, all_token_ids, args.output, args.cjk_categorize)


if __name__ == "__main__":
    main()

import json

def analyze_categories(json_file):
    """
    Analyzes categorized tokens and writes encountered/not encountered characters to files.
    """

    with open(json_file, 'r', encoding='utf-8') as f:
        categories = json.load(f)

    total_tokens = 0
    for category_tokens in categories.values():
        total_tokens += len(category_tokens)

    print("Token Category Statistics:")
    for category, tokens in categories.items():
        count = len(tokens)
        percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"- {category.capitalize()}: {count} ({percentage:.2f}%)")

    # --- Comprehensive Chinese Character Analysis ---

    cjk_ranges = [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
        (0x2CEB0, 0x2EBEF), # CJK Unified Ideographs Extension F
        (0x30000, 0x3134F),  # CJK Unified Ideographs Extension G
        (0x31350, 0x323AF)   # CJK Unified Ideographs Extension H
        ]

    all_possible_chinese_chars = set()
    for start, end in cjk_ranges:
        for char_code in range(start, end + 1):
            all_possible_chinese_chars.add(chr(char_code))

    initial_possible_chars = len(all_possible_chinese_chars)

    encountered_chinese_chars = set()
    if "chinese" in categories:
        category = "chinese"
    else:
        category = "vocab"

    for token_data in categories[category]:
        for char in token_data["token"]:
            if any(start <= ord(char) <= end for start, end in cjk_ranges):
                encountered_chinese_chars.add(char)

    remaining_chinese_chars = all_possible_chinese_chars - encountered_chinese_chars
    remaining_count = len(remaining_chinese_chars)
    percentage_remaining = (remaining_count / initial_possible_chars) * 100 if initial_possible_chars > 0 else 0
    encountered_count = len(encountered_chinese_chars)
    percentage_encountered = (encountered_count / initial_possible_chars) * 100 if initial_possible_chars > 0 else 0

    print("\nComprehensive Chinese Character Analysis:")
    print(f"- Total possible unique CJK Unified Ideographs (including extensions): {initial_possible_chars}")
    print(f"- Number of unique CJK characters encountered in tokens: {encountered_count} ({percentage_encountered:.4f}%)")
    print(f"- Number of unique CJK characters NOT encountered: {remaining_count}")
    print(f"- Percentage of unique CJK characters NOT encountered: {percentage_remaining:.4f}%")


    # --- Korean Character Analysis ---

    hangul_start = 0xAC00
    hangul_end = 0xD7A3

    all_possible_korean_chars = set()
    for char_code in range(hangul_start, hangul_end + 1):
        all_possible_korean_chars.add(chr(char_code))

    initial_possible_korean_chars = len(all_possible_korean_chars)
    encountered_korean_chars = set()

    if "korean" in categories:
        category = "korean"
    else:
        category = "vocab"
    for token_data in categories[category]:
        for char in token_data["token"]:
            if hangul_start <= ord(char) <= hangul_end:
                encountered_korean_chars.add(char)

    remaining_korean_chars = all_possible_korean_chars - encountered_korean_chars
    remaining_korean_count = len(remaining_korean_chars)
    percentage_korean_remaining = (remaining_korean_count / initial_possible_korean_chars) * 100 if initial_possible_korean_chars > 0 else 0

    encountered_korean_count = len(encountered_korean_chars)
    korean_encountered_percentage = (encountered_korean_count / initial_possible_korean_chars) * 100 if initial_possible_korean_chars > 0 else 0


    print("\nKorean Character Analysis:")
    print(f"- Total possible unique Hangul Syllables: {initial_possible_korean_chars}")
    print(f"- Number of unique Hangul Syllables encountered: {encountered_korean_count} ({korean_encountered_percentage:.4f}%)")
    print(f"- Number of unique Hangul Syllables NOT encountered: {remaining_korean_count}")
    print(f"- Percentage of unique Hangul Syllables NOT encountered: {percentage_korean_remaining:.4f}%")



    # --- Write characters to files ---
    with open("chinese_encountered.txt", "w", encoding="utf-8") as f:
        for char in sorted(list(encountered_chinese_chars)):  # Sorted for consistency
            f.write(char + "\n")

    with open("chinese_not_encountered.txt", "w", encoding="utf-8") as f:
        for char in sorted(list(remaining_chinese_chars)):
            f.write(char + "\n")

    with open("korean_encountered.txt", "w", encoding="utf-8") as f:
        for char in sorted(list(encountered_korean_chars)):
            f.write(char + "\n")

    with open("korean_not_encountered.txt", "w", encoding="utf-8") as f:
        for char in sorted(list(remaining_korean_chars)):
            f.write(char + "\n")

    print("\nCharacter lists written to files.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze categorized token JSON file.")
    parser.add_argument("json_file", help="Path to the JSON file to analyze.")
    args = parser.parse_args()

    analyze_categories(args.json_file)


if __name__ == "__main__":
    main()

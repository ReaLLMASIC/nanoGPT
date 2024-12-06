from transformers import AutoTokenizer
from rich.table import Table
from rich.console import Console
import tiktoken

console = Console()

# Define the Unicode ranges for CJK and subcategories
unicode_ranges = {
    "C": {
        "Unified Ideographs": [(0x4E00, 0x9FFF)],
        "Extension A": [(0x3400, 0x4DBF)],
        "Extension B": [(0x20000, 0x2A6DF)],
        "Extension C": [(0x2A700, 0x2B73F)],
        "Extension D": [(0x2B740, 0x2B81F)],
        "Extension E": [(0x2B820, 0x2CEAF)],
        "Compatibility Ideographs": [(0xF900, 0xFAFF)],
    },
    "J": {
        "Hiragana": [(0x3040, 0x309F)],
        "Katakana": [(0x30A0, 0x30FF)],
        "Katakana Phonetic Extensions": [(0x31F0, 0x31FF)],
        "Kanbun": [(0x3190, 0x319F)],
    },
    "K": {
        "Hangul Syllables": [(0xAC00, 0xD7AF)],
        "Hangul Jamo": [(0x1100, 0x11FF)],
        "Hangul Jamo Extended-A": [(0xA960, 0xA97F)],
        "Hangul Jamo Extended-B": [(0xD7B0, 0xD7FF)],
        "Compatibility Jamo": [(0x3130, 0x318F)],
    },
}

def is_in_ranges(char, ranges):
    """Check if a character is within any of the given Unicode ranges."""
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in ranges)

def count_range_size(ranges):
    """Calculate how many characters are possible within the given ranges."""
    return sum((end - start + 1) for start, end in ranges)

def count_tokens_containing_range(vocab_list, ranges):
    """Count how many tokens contain at least one character in the given ranges."""
    count = 0
    for token in vocab_list:
        if any(is_in_ranges(ch, ranges) for ch in token):
            count += 1
    return count

def count_unique_characters(vocab_list, ranges):
    """Count how many unique characters from the given ranges appear in the vocab."""
    unique_chars = set()
    for token in vocab_list:
        for char in token:
            if is_in_ranges(char, ranges):
                unique_chars.add(char)
    return len(unique_chars)

def count_total_cjk_tokens(vocab_list):
    """Count total tokens that belong to any C, J, or K category without double-counting."""
    matched_tokens = set()
    for token in vocab_list:
        for lang, sub_ranges in unicode_ranges.items():
            for subcat, ranges in sub_ranges.items():
                if any(is_in_ranges(ch, ranges) for ch in token):
                    matched_tokens.add(token)
                    break
    return len(matched_tokens)

def analyze_tokenizer(tokenizer_name):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_list = list(tokenizer.get_vocab().keys())

    total_tokens = len(vocab_list)

    console.print(f"# Tokenizer Analysis: {tokenizer_name}\n", style="bold cyan")

    # Table for counting tokens and symbol coverage
    token_count_table = Table(title="Subcategory Analysis")
    token_count_table.add_column("Language", justify="left")
    token_count_table.add_column("Subcategory", justify="left")
    token_count_table.add_column("Token Count", justify="right")
    token_count_table.add_column("Unique Symbols Found", justify="right")
    token_count_table.add_column("Total Possible Characters", justify="right")
    token_count_table.add_column("% of Symbols in Range", justify="right")

    total_unique_symbols = 0
    total_possible_symbols = 0
    total_percentage_sum = 0
    row_count = 0

    for lang, sub_ranges in unicode_ranges.items():
        for subcat, ranges in sub_ranges.items():
            token_count = count_tokens_containing_range(vocab_list, ranges)
            unique_count = count_unique_characters(vocab_list, ranges)
            total_possible = count_range_size(ranges)
            percentage = (unique_count / total_possible * 100) if total_possible > 0 else 0.0

            # Update totals
            total_unique_symbols += unique_count
            total_possible_symbols += total_possible
            total_percentage_sum += percentage
            row_count += 1

            token_count_table.add_row(
                lang,
                subcat,
                str(token_count),
                str(unique_count),
                str(total_possible),
                f"{percentage:.2f}%"
            )

    # Calculate total tokens in any C, J, or K category
    total_cjk_tokens = count_total_cjk_tokens(vocab_list)
    avg_percentage = total_unique_symbols / total_possible_symbols * 100

    # Add total row
    token_count_table.add_row(
        "TOTAL (Any C, J, K)",
        "",
        str(total_cjk_tokens),
        str(total_unique_symbols),
        str(total_possible_symbols),
        f"{avg_percentage:.2f}%",
        style="bold"
    )
    console.print(token_count_table)

    console.print(f"Total Tokens: {total_tokens}", style="bold yellow")

# Compare specific tokenizers
tokenizers = [
    {"name": "google/gemma-7b"},
    {"name": "mistralai/Mistral-7B-Instruct-v0.3"},
]

for t in tokenizers:
    analyze_tokenizer(t["name"])


#!/usr/bin/env python3
import json
import sys
from rich.console import Console
from rich.table import Table

def process_json(file_path):
    # Load the JSON file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # Extract tokens from the three sources:
    # 1. added_tokens (top level)
    added_tokens = data.get("added_tokens", [])
    # 2. model.vocab and 3. model.merges (inside the "model" object)
    model = data.get("model", {})
    # Assume vocab and merges are already in key–value format (i.e. dictionaries)
    vocab = model.get("vocab", {}) if isinstance(model.get("vocab"), dict) else {}
    merges = model.get("merges", {}) if isinstance(model.get("merges"), dict) else {}

    # Transform added_tokens:
    # Create a dictionary mapping each token's content to its id.
    transformed_added_tokens = {}
    for token in added_tokens:
        if isinstance(token, dict) and "content" in token and "id" in token:
            transformed_added_tokens[token["content"]] = token["id"]

    # Compute counts for each category.
    count_added = len(transformed_added_tokens)
    count_vocab = len(vocab)
    count_merges = len(merges)
    total = count_added + count_vocab + count_merges
    total = total if total > 0 else 1  # avoid division by zero

    # Compute percentages.
    pct_added = (count_added / total) * 100
    pct_vocab = (count_vocab / total) * 100
    pct_merges = (count_merges / total) * 100

    # Display a rich table with the stats.
    console = Console()
    table = Table(title="Token Statistics")

    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta", justify="right")
    table.add_column("Percentage", style="green", justify="right")

    table.add_row("added_tokens", str(count_added), f"{pct_added:.2f}%")
    table.add_row("vocab", str(count_vocab), f"{pct_vocab:.2f}%")
    table.add_row("merges", str(count_merges), f"{pct_merges:.2f}%")

    console.print(table)

    # Create a flattened structure (a single dict) containing all tokens.
    # The output will look like:
    # {
    #    "õ": 32000,
    #    "another_token": 32001,
    #    ...
    # }
    flattened_tokens = {}
    flattened_tokens.update(transformed_added_tokens)
    flattened_tokens.update(vocab)
    flattened_tokens.update(merges)

    # Export the flattened tokens to a new JSON file.
    output_file = "deepseek_tokens.josn"  # using the filename as specified
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(flattened_tokens, out_f, indent=2, ensure_ascii=False)
        console.print(f"[bold green]Successfully exported flattened tokens to {output_file}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error exporting data: {e}[/bold red]")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <json_file>")
        sys.exit(1)
    process_json(sys.argv[1])

if __name__ == "__main__":
    main()


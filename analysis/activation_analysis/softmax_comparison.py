import argparse
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.style import Style

console = Console()

def numpy_softmax(x):
    e_x = np.exp(x - np.max(x))
    values = e_x
    normalized = e_x / e_x.sum()
    return values, normalized

def obo(x):
    e_x = np.exp(x - np.max(x))
    obo_1 = e_x / (1 + e_x.sum())
    obo_10 = e_x / (10 + e_x.sum())
    return obo_1, obo_10

def format_percentage(value):
    if value == "NaN": return value
    value = float(value)
    return f"[green]+{value:.2f}%[/green]" if value > 0 else f"[red]{value:.2f}%[/red]" if value < 0 else f"{value:.2f}%"

def format_ratio(value):
    if value == "NaN": return value
    value = float(value)
    return f"[green]{value:.2f}[/green]" if value > 1.0001 else f"[red]{value:.2f}[/red]" if value < 0.9999 else f"{value:.2f}"

def vector_similarity(x, y):
    x_norm = x / (np.linalg.norm(x) + 1e-16)
    y_norm = y / (np.linalg.norm(y) + 1e-16)
    return np.dot(x_norm, y_norm)

def magnitude_change_ratio(x, y):
    x_mag, y_mag = np.linalg.norm(x), np.linalg.norm(y)
    return y_mag / (x_mag + 1e-16) if x_mag else "NaN"

def sum_ratio(x, y):
    return np.sum(x) / np.sum(y) if np.sum(y) else "NaN"

def create_table(title, color, columns, rows):
    table = Table(title=title, style=Style(color=color))
    for col, justify in columns:
        table.add_column(col, justify=justify)
    for row in rows:
        table.add_row(*row)
    console.print(table)

def compute_changes(base, scaled):
    changes = []
    for b, s in zip(base, scaled):
        if b == 0:
            changes.append("NaN")
        else:
            changes.append(f"{(s - b) / b * 100:.2f}")
    return [format_percentage(c) for c in changes]

def apply_torch_fn(fn, x):
    values = fn(torch.tensor(x, dtype=torch.float32)).numpy()
    total = values.sum()
    normalized = values / total if total else np.zeros_like(values)
    return values, normalized

def main():
    parser = argparse.ArgumentParser(description="Compare activation function results.")
    parser.add_argument("numbers", nargs="+", type=float)
    parser.add_argument("--scale", type=float, default=10.0)
    args = parser.parse_args()

    numbers = np.array(args.numbers)
    scaled_numbers = numbers * args.scale

    pytorch_fns = [
        ("ReLU", torch.nn.ReLU()),
        ("Softplus", torch.nn.Softplus()),
        ("Sigmoid", torch.nn.Sigmoid()),
    ]

    methods = {"Softmax": numpy_softmax, "Obo": obo}
    for name, module in pytorch_fns:
        methods[name] = lambda x, m=module: apply_torch_fn(m, x)

    results = {}
    for name, func in methods.items():
        results[name] = {
            'base': func(numbers),
            'scaled': func(scaled_numbers)
        }

    # Unscaled and Scaled Results Tables
    for scale_label, data, color in [("Unscaled Results", numbers, "cyan"), ("Scaled Results", scaled_numbers, "magenta")]:
        result_type = 'base' if scale_label == 'Unscaled Results' else 'scaled'
        rows = []
        for i, num in enumerate(data):
            row = [f"{num:.2f}"]
            for name in methods:
                values, normalized = results[name][result_type]
                row.extend([f"{values[i]:.2f}", f"{normalized[i]:.2f}"])
            rows.append(row)

        columns = [("Input", "left")]
        for name in methods:
            columns.extend([(name, "right"), (f"Normalized {name}", "right")])

        create_table(scale_label, color, columns, rows)

    # Percentage Changes Table
    pct_rows = []
    for i in range(len(numbers)):
        row = [f"{numbers[i]:.2f}"]
        for name in methods:
            base_values, base_norm = results[name]['base']
            scaled_values, scaled_norm = results[name]['scaled']
            row.extend(compute_changes(base_values[i:i+1], scaled_values[i:i+1]))
            row.extend(compute_changes(base_norm[i:i+1], scaled_norm[i:i+1]))
        pct_rows.append(row)

    pct_columns = [("Input", "left")]
    for name in methods:
        pct_columns.extend([(f"{name}", "right"), (f"Normalized {name}", "right")])

    create_table("Percentage Changes", "green", pct_columns, pct_rows)

    # Vector Similarity and Magnitude Change Table
    sim_rows = []
    for name in methods:
        base_values, base_norm = results[name]['base']
        scaled_values, scaled_norm = results[name]['scaled']
        sim_rows.append([
            name,
            f"{vector_similarity(base_values, scaled_values):.4f}",
            format_ratio(magnitude_change_ratio(base_values, scaled_values)),
            format_ratio(sum_ratio(scaled_values, base_values))
        ])
        sim_rows.append([
            f"Normalized {name}",
            f"{vector_similarity(base_norm, scaled_norm):.4f}",
            format_ratio(magnitude_change_ratio(base_norm, scaled_norm)),
            format_ratio(sum_ratio(scaled_norm, base_norm))
        ])

    create_table("Vector Similarity and Magnitude Change", "blue",
                 [("Metric", "left"), ("Dot Product", "right"), ("Magnitude Ratio", "right"), ("Sum Ratio", "right")],
                 sim_rows)

if __name__ == "__main__":
    main()

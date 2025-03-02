import os
import sys
import argparse
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add top level dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Model Report Generator")
    parser.add_argument("ckpt_path", help="Path to the checkpoint file")
    parser.add_argument('--bins', type=int, default=20, help='Number of bins for histograms')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on')
    return parser.parse_args()

def load_model(ckpt_path, device):
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint.get('model_args', None)
    if model_args is None:
        sys.exit("Model arguments not found in checkpoint.")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def compute_stats(tensor):
    tensor_np = tensor.detach().cpu().numpy()
    flat_tensor = tensor_np.flatten()
    stats = {
        'shape': tensor_np.shape,
        'min': np.min(flat_tensor),
        'max': np.max(flat_tensor),
        'q1': np.percentile(flat_tensor, 25),
        'median': np.median(flat_tensor),
        'q3': np.percentile(flat_tensor, 75),
        'mean': np.mean(flat_tensor),
        'zeros_percent': np.sum(flat_tensor == 0) / flat_tensor.size * 100
    }
    return stats

def create_histogram(tensor, full_key, bins, report_dir):
    # Compute statistics
    stats = compute_stats(tensor)

    # Convert tensor to NumPy array for plotting
    tensor_np = tensor.detach().cpu().numpy().flatten()

    # Create the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(tensor_np, bins=bins, kde=False)
    plt.title(f"Histogram of {full_key}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Overlay statistics
    stats_text = '\n'.join([
        f"Shape: {stats['shape']}",
        f"Min: {stats['min']:.6f}",
        f"Max: {stats['max']:.6f}",
        f"Mean: {stats['mean']:.6f}",
        f"Median: {stats['median']:.6f}",
        f"Q1: {stats['q1']:.6f}",
        f"Q3: {stats['q3']:.6f}",
        f"Zeros (%): {stats['zeros_percent']:.2f}%"
    ])
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Save the figure
    filename = full_key.replace('.', '_').replace('/', '_')
    image_path = os.path.join(report_dir, f"{filename}_histogram.png")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return image_path

def create_heatmap(tensor, full_key, report_dir):
    if tensor.ndim != 2:
        return None  # Only create heatmap for 2D tensors

    # Compute statistics
    stats = compute_stats(tensor)

    # Convert tensor to NumPy array for plotting
    tensor_np = tensor.detach().cpu().numpy()

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(tensor_np, cmap='viridis')
    plt.title(f"Heatmap of {full_key}")

    # Overlay statistics
    stats_text = '\n'.join([
        f"Shape: {stats['shape']}",
        f"Min: {stats['min']:.6f}",
        f"Max: {stats['max']:.6f}",
        f"Mean: {stats['mean']:.6f}",
        f"Median: {stats['median']:.6f}",
        f"Q1: {stats['q1']:.6f}",
        f"Q3: {stats['q3']:.6f}",
        f"Zeros (%): {stats['zeros_percent']:.2f}%"
    ])
    plt.annotate(stats_text, xy=(1.05, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Save the figure
    filename = full_key.replace('.', '_').replace('/', '_')
    image_path = os.path.join(report_dir, f"{filename}_heatmap.png")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return image_path

def generate_report(model, bins, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    state_dict = model.state_dict()

    # Create a PDF report
    pdf_path = os.path.join(report_dir, 'model_report.pdf')
    with PdfPages(pdf_path) as pdf:
        for name, param in state_dict.items():
            print(f"Processing {name}...")
            # Compute statistics
            stats = compute_stats(param)

            # Generate histogram
            hist_image_path = create_histogram(param, name, bins, report_dir)

            # Generate heatmap if tensor is 2D
            heatmap_image_path = None
            if param.ndim == 2:
                heatmap_image_path = create_heatmap(param, name, report_dir)

            # Add histogram to PDF
            hist_image = plt.imread(hist_image_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(hist_image)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            # Add heatmap to PDF if available
            if heatmap_image_path:
                heatmap_image = plt.imread(heatmap_image_path)
                plt.figure(figsize=(10, 8))
                plt.imshow(heatmap_image)
                plt.axis('off')
                pdf.savefig()
                plt.close()

    print(f"\nReport generated at {pdf_path}")

def main():
    args = parse_args()

    model = load_model(args.ckpt_path, args.device)
    report_dir = os.path.join('checkpoint_analysis', 'report')
    generate_report(model, args.bins, report_dir)

if __name__ == '__main__':
    main()


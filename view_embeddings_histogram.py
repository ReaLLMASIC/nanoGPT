import numpy as np
import argparse
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_histograms(data, output_path):
    # Create a histogram for each column
    num_columns = data.shape[1]
    num_rows = int(np.ceil(num_columns / 4))  # Adjust rows based on the number of columns
    plt.figure(figsize=(20, num_rows * 5))  # Adjust height based on the number of rows
    
    for i in range(num_columns):
        plt.subplot(num_rows, 4, i+1)
        plt.hist(data[:, i], bins=30, alpha=0.7)
        plt.title(f'Column {i}')
    
    plt.tight_layout()
    plt.savefig(output_path + '_histograms.png')
    plt.show()

def plot_3d_histogram_columns(data, output_path):
    # Create a 3D histogram along columns
    x, y, z = [], [], []
    
    for i in range(data.shape[1]):
        hist, bin_edges = np.histogram(data[:, i], bins=30)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        x.extend([i] * len(hist))
        y.extend(bin_centers)
        z.extend(hist)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=5, color=z, colorscale='Viridis', opacity=0.8)
    )])
    
    fig.update_layout(
        title="3D Histogram of Embeddings (Columns)",
        scene=dict(
            xaxis_title="Column",
            yaxis_title="Value",
            zaxis_title="Count"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.write_html(output_path + '_3d_histogram_columns.html')
    fig.show()

def plot_3d_histogram_rows(data, output_path):
    # Create a 3D histogram along rows
    x, y, z = [], [], []
    
    for i in range(data.shape[0]):
        hist, bin_edges = np.histogram(data[i, :], bins=30)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        x.extend([i] * len(hist))
        y.extend(bin_centers)
        z.extend(hist)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=5, color=z, colorscale='Viridis', opacity=0.8)
    )])
    
    fig.update_layout(
        title="3D Histogram of Embeddings (Rows)",
        scene=dict(
            xaxis_title="Row",
            yaxis_title="Value",
            zaxis_title="Count"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.write_html(output_path + '_3d_histogram_rows.html')
    fig.show()

def main():
    parser = argparse.ArgumentParser(description="View histograms of embeddings from a .npy file.")
    parser.add_argument('--npy_path', type=str, required=True, help="Path to the .npy file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output plots.")
    args = parser.parse_args()
    
    data = np.load(args.npy_path)
    
    plot_histograms(data, args.output_path)
    plot_3d_histogram_columns(data, args.output_path)
    plot_3d_histogram_rows(data, args.output_path)

if __name__ == "__main__":
    main()


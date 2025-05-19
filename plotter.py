import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import os
import sys

matplotlib.use('Agg')

# Function: Load Data from CSV
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df['Actual Values'], df['Predicted Values']

# Function: Extract Title and PNG Name
def extract_title_and_filename(csv_file):
    filename = os.path.splitext(os.path.basename(csv_file))[0]
    title = filename.replace('_', ' ').title()
    png_name = f"{filename}.png"
    return title, png_name

# Function: Create 2D Histogram Plot
def plot_2d_histogram(sample_targets, predictions, title, png_name, vmin=1, vmax=1e5):
    bins = np.logspace(-4, 5, 200)
    hist2d, x_edges, y_edges = np.histogram2d(sample_targets, predictions, bins=(bins, bins))

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.pcolormesh(x_edges, y_edges, hist2d.T, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(cax, ax=ax, label='Number of Points')

    ax.set_xlim(1e-4, 10)
    ax.set_ylim(1e-4, 10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Actual Values (Log Scale)')
    ax.set_ylabel('Predicted Values (Log Scale)')
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(png_name, dpi=300)

# Main Script
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    sample_targets, predictions = load_data(csv_file)
    title, png_name = extract_title_and_filename(csv_file)

    # Setting consistent color scale for comparison
    plot_2d_histogram(sample_targets, predictions, title, png_name, vmin=1, vmax=1e5)

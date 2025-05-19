import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import os
import sys

matplotlib.use('Agg')

# Function: Load Data from CSV
def load_loss_data(csv_file):
    df = pd.read_csv(csv_file)
    return df['epoch'], df['train_loss'], df['val_loss']

# Function: Extract Title and PNG Name
def extract_title_and_filename(csv_file):
    filename = os.path.splitext(os.path.basename(csv_file))[0]
    title = filename.replace('_', ' ').title()
    png_name = f"{filename}.png"
    return title, png_name

# Function: Create 2D Histogram Plot
def plot_loss(epoch, train_loss, val_loss, title, png_name):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epoch, train_loss, label='Training Loss', linestyle='-', marker='o')
    ax.plot(epoch, val_loss, label='Validation Loss', linestyle='-', marker='o')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.semilogx(True)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(png_name, dpi=300)
    plt.show()

# Main Script
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    epoch, train_loss, val_loss = load_loss_data(csv_file)
    title, png_name = extract_title_and_filename(csv_file)

    # Setting consistent color scale for comparison
    plot_loss(epoch, train_loss, val_loss, title, png_name)

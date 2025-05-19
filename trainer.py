import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import uproot
import argparse
import yaml
import csv
from scipy.special import erfinv
from scipy.stats import rankdata
import os
from datetime import datetime

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Standard Model Configuration
TRENAME = "mcmc"
BRANCHES = [
    "chi10", "chi20", "chi1pm", "chi2pm", "g", "M1", "M2", "mu",
    "dL", "dR", "uL", "uR", "t1", "b1", "chi1pm_ctau", "chi20_ctau",
    "g_ctau", "eR_ctau", "Likelihood", "bf_cms_sus_19_006_mu1p0f",
    "bf_cms_sus_20_001_mu1p0s", "bf_cms_sus_21_006_mu1p0f",
    "bf_cms_sus_18_004_mu1p0f", "bf_cms_sus_21_007_mb_mu1p0s"
]

# Function to load data
def load_data(filename):
    with uproot.open(filename) as file:
        tree = file[TRENAME]
        data = tree.arrays(BRANCHES, library="pd")
    return data

# Gaussian Rank Transformation
def gaussian_rank(arr):
    ranked = rankdata(arr, method='average')
    fractional_ranks = (ranked - 0.5) / len(arr)
    return np.maximum(np.sqrt(2) * erfinv(2 * fractional_ranks - 1), 10**-4)

# Feature Engineering with Gaussian Rank Transformations
def feature_engineering(data):
    mass_features = ['chi10', 'chi20', 'chi1pm', 'chi2pm', 'g', 'M1', 'M2', 'mu']
    for feature in mass_features:
        data[f'gaussian_rank_{feature}'] = gaussian_rank(data[feature])

    data['min_squark'] = data[['dL', 'dR', 'uL', 'uR']].min(axis=1)
    data['min_squark_minus_chi10'] = np.log10(np.maximum(abs(data['min_squark'] - data['chi10'].abs()), 10**-4))
    data['min_t1_b1_minus_chi10'] = np.log10(np.maximum(abs(data[['t1', 'b1']].min(axis=1) - data['chi10'].abs()), 10**-4))
    data['chi1pm_minus_chi10'] = np.log10(np.maximum(abs(data['chi1pm'] - data['chi10'].abs()), 10**-4))
    data['chi1pm_minus_chi20'] = np.log10(np.maximum(abs(data['chi1pm'] - data['chi20'].abs()), 10**-4))
    data['chi20_minus_chi10'] = np.log10(np.maximum(abs(data['chi20'].abs() - data['chi10'].abs()), 10**-4))

    data['combined_bf'] = data.filter(regex='bf_').prod(axis=1)

    input_features = [f'gaussian_rank_{f}' for f in mass_features] + [
        'min_squark_minus_chi10', 'min_t1_b1_minus_chi10', 'chi1pm_minus_chi10',
        'chi1pm_minus_chi20', 'chi20_minus_chi10'
    ]

    return data, input_features

# Stable Model with Clamped Activation
class StableModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Softplus()
        )

    def forward(self, x):
        return torch.clamp(self.model(x), min=1e-4, max=1e6)

# Stable Log-Cosh Loss
class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        diff = torch.clamp(pred - target, -20, 20)
        return torch.mean(torch.log(torch.cosh(diff)))

# Training Function with CSV Logging and Model Saving
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, config, date_str):
    model_filename = f"model_batchSize{config['batch_size']}_learningRate{config['learning_rate']}_epochs{config['num_epochs']}_{date_str}.pt"
    loss_filename = f"loss_batchSize{config['batch_size']}_learningRate{config['learning_rate']}_epochs{config['num_epochs']}_{date_str}.csv"

    best_val_loss = float('inf')

    with open(loss_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])

        for epoch in range(config['num_epochs']):
            model.train()
            train_loss = 0.0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            writer.writerow([epoch + 1, train_loss, val_loss])
            print(f"Epoch {epoch + 1}/{config['num_epochs']} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_filename)

# Evaluation and Saving Function
def evaluate_and_save_model(model, X_val_tensor, y_val_tensor, config, date_str):
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_tensor).cpu().numpy().flatten()
        actuals = y_val_tensor.cpu().numpy().flatten()
    filename = f"predictions_batchSize{config['batch_size']}_learningRate{config['learning_rate']}_epochs{config['num_epochs']}_{date_str}.csv"

    # Create DataFrame with predictions and actuals
    results = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})

    # Construct filename using the model name
    results.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

# Main Function
def main():
    date_str = datetime.now().strftime('%Y%m%d')
    parser = argparse.ArgumentParser(description='Train StableModel')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the ROOT file')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    data = load_data(args.input_file)
    data, input_features = feature_engineering(data)

    X = data[input_features].values
    y = data[['combined_bf']].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = StableModel(X_train.shape[1])

    criterion = LogCoshLoss()
    learning_rate = float(config['learning_rate'])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config['scheduler_T0'])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                                                              torch.tensor(y_train, dtype=torch.float32)), 
                                                                              batch_size=config['batch_size'], shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                                                            torch.tensor(y_val, dtype=torch.float32)), 
                                                                            batch_size=config['batch_size'])

    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, config, date_str)
    evaluate_and_save_model(model, torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32), config, date_str)

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import uproot
import argparse
import yaml
import csv
from scipy.special import erfinv
from scipy.stats import rankdata
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

def load_data(filename):
    with uproot.open(filename) as file:
        tree = file[TRENAME]
        data = tree.arrays(BRANCHES, library="pd")
    return data

def gaussian_rank(arr):
    ranked = rankdata(arr, method='average')
    fractional_ranks = (ranked - 0.5) / len(arr)
    return np.maximum(np.sqrt(2) * erfinv(2 * fractional_ranks - 1), 10**-4)

def feature_engineering(data):
    mass_features = ['chi10', 'chi20', 'chi1pm', 'chi2pm', 'g', 'M1', 'M2', 'mu']
    for feature in mass_features:
        data[f'gaussian_rank_{feature}'] = gaussian_rank(data[feature])

    data['min_squark'] = data[['dL', 'dR', 'uL', 'uR']].min(axis=1)
    data['min_squark_minus_chi10'] = np.log10(data['min_squark'] - data['chi10'].abs())
    data['min_t1_b1_minus_chi10'] = np.log10(data[['t1', 'b1']].min(axis=1) - data['chi10'].abs())
    data['chi1pm_minus_chi10'] = np.log10(data['chi1pm'] - data['chi10'].abs())
    data['chi1pm_minus_chi10'] = np.log10(data['chi1pm'] - data['chi10'].abs())
    data['chi20_minus_chi10'] = np.log10(data['chi20'].abs() - data['chi10'].abs())
    data['g_minus_chi10'] = np.log10(data['g'].abs() - data['chi10'].abs())

    data['combined_bf'] = data.filter(regex='bf_').prod(axis=1)

    input_features = [f'gaussian_rank_{f}' for f in mass_features] + [
        'min_squark_minus_chi10', 'min_t1_b1_minus_chi10', 'chi1pm_minus_chi10',
        'chi1pm_minus_chi10', 'chi20_minus_chi10','g_minus_chi10'
    ]

    return data, input_features


class SELUNetwork(nn.Module):
    def __init__(self, input_dim=27, output_dim=1, hidden_dim=256, depth=8, dropout_rate=0.1):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SELU())
        layers.append(nn.AlphaDropout(dropout_rate))

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SELU())
            layers.append(nn.AlphaDropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')  # SELU-compatible
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


class RelativeErrorLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.abs((pred - target) / (target + self.eps)))

    
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
                scheduler.step()
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

def evaluate_and_save_model(model, X_val_tensor, y_val_tensor, config, date_str):
    model.eval()
    with torch.no_grad():
        predictions = 10 ** model(X_val_tensor).cpu().numpy().flatten()
        actuals = 10 ** y_val_tensor.cpu().numpy().flatten()
    filename = f"predictions_batchSize{config['batch_size']}_learningRate{config['learning_rate']}_epochs{config['num_epochs']}_{date_str}.csv"
    results = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})
    results.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def main():
    date_str = datetime.now().strftime('%Y%m%d')
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the ROOT file')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    data = load_data(args.input_file)
    data, input_features = feature_engineering(data)
    data = data[data['combined_bf'] > 0]

    X = data[input_features].values
    y = np.log10(data[['combined_bf']].values)


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SELUNetwork(input_dim=X_train.shape[1], output_dim=1)

    criterion = RelativeErrorLoss()
    learning_rate = float(config['learning_rate'])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(config['weight_decay']))
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=config['num_epochs'],
                           steps_per_epoch=len(X_train) // config['batch_size'] + 1, pct_start=0.1, anneal_strategy='cos')

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=config['batch_size'], shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=config['batch_size'])

    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, config, date_str)
    evaluate_and_save_model(model, torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32), config, date_str)

if __name__ == '__main__':
    main()
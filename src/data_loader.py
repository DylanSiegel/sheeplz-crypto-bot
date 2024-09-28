# src/data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading data.
    """

    def __init__(self, features: pd.DataFrame, targets: pd.Series):
        self.X = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(targets.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Example usage
# dataset = TradingDataset(X_selected, target)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

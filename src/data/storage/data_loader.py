# src/data/storage/data_loader.py

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import torch

class TradingDataset(Dataset):
    def __init__(self, features: pd.DataFrame, targets: pd.Series):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

def create_data_loader(features: pd.DataFrame, targets: pd.Series, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TradingDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
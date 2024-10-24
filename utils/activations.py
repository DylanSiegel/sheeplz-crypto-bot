# File: utils/activations.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class ModifiedReLU(nn.Module):
    """Modified ReLU that maintains normalization properties"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) * (1 - torch.exp(-x))

def get_activation_fn(name: str) -> Callable:
    """Get activation function by name"""
    activations = {
        'tanh': torch.tanh,
        'modified_relu': ModifiedReLU(),
        'none': lambda x: x,
    }

    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")

    return activations[name]

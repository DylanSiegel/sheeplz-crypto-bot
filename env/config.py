# env/config.py
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class EnvironmentConfig:
    """
    Production environment config for a futures trading environment with dynamic leverage.
    """

    # Environment Basic
    lookback_window: int = 30
    initial_balance: float = 100_000.0
    max_leverage: float = 5.0
    reward_scaling: float = 1.0
    max_steps: int = 2000
    historical_data_path: str = "data/btc_usdt_1min.parquet"
    buffer_capacity: int = 100_000

    # Observations & Actions
    action_dim: int = 4
    state_dim: int = 0  # Will be dynamically assigned in environment
    hidden_dim: int = 256  # Larger for production-level capacity
    time_encoding_dim: int = 16
    attention_dim: int = 32
    num_market_modes: int = 3

    # Core Hyperparams
    alpha: float = 0.2
    meta_lr: float = 1e-4
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    max_grad_norm: float = 1.0

    # Exploration & Noise
    noise_std: float = 0.1
    epsilon: float = 1e-8

    # Meta
    meta_input_dim: int = 6  # E.g. if you want an extra dimension
    num_hyperparams: int = 8

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

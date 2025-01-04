# config.py
import torch
from dataclasses import dataclass

@dataclass
class EnvironmentConfig:
    """Configuration for environment, training, and hyperparameters."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    action_dim: int = 2
    state_dim: int = 8
    hidden_dim: int = 256
    time_encoding_dim: int = 16
    attention_dim: int = 64
    num_mlp_layers: int = 2
    dropout_rate: float = 0.1
    lr: float = 3e-4
    meta_lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    target_entropy_factor: float = 1.0
    epsilon: float = 1e-6
    max_grad_norm: float = 5.0
    buffer_capacity: int = 100000
    batch_size: int = 64
    meta_input_dim: int = 10
    num_hyperparams: int = 5
    num_market_modes: int = 3
    window_size: int = 10
    custom_layers: list = None
    add_search_to_replay: bool = False
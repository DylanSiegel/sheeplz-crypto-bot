import torch
from dataclasses import dataclass

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class MetaSACConfig:
    """Configuration for MetaSAC agent and networks."""
    state_dim: int
    action_dim: int
    num_hyperparams: int = 5
    hidden_dim: int = 64
    attention_dim: int = 10
    meta_input_dim: int = 5
    time_encoding_dim: int = 10
    num_mlp_layers: int = 3
    dropout_rate: float = 0.1
    lr: float = 1e-3
    meta_lr: float = 1e-4
    alpha: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    max_grad_norm: float = 1.0
    epsilon: float = 1e-10
    device: torch.device = device
    replay_buffer_capacity: int = 1000000
    window_size: int = 10
    custom_layers: list = None

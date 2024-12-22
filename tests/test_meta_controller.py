# tests/test_meta_controller.py
import pytest
import torch
from networks import MetaController
from dataclasses import dataclass

@dataclass
class Config:
    meta_input_dim: int = 4
    hidden_dim: int = 32
    num_hyperparams: int = 5
    num_mlp_layers: int = 3
    dropout_rate: float = 0.1
    window_size: int = 10
    device: torch.device = torch.device("cpu")  # CHANGED: Added device attribute

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_meta_controller(device):
    config = Config()
    batch_size = 5
    controller = MetaController(config).to(device)
    meta_input_tensor = torch.randn(batch_size, config.meta_input_dim).to(device)
    reward_stats_tensor = torch.randn(batch_size, 2).to(device)
    lr_actor, lr_critic, lr_alpha, tau, gamma = controller(meta_input_tensor, reward_stats_tensor)
    assert lr_actor.shape == (batch_size,)
    assert lr_critic.shape == (batch_size,)
    assert lr_alpha.shape == (batch_size,)
    assert tau.shape == (batch_size,)
    assert gamma.shape == (batch_size,)
    assert not torch.isnan(lr_actor).any()
    assert not torch.isnan(lr_critic).any()
    assert not torch.isnan(lr_alpha).any()
    assert not torch.isnan(tau).any()
    assert not torch.isnan(gamma).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_meta_controller_edge_cases(device):
    config = Config()
    batch_size = 5
    controller = MetaController(config).to(device)

    # Test with NaN values in meta_input
    meta_input_tensor_nan = torch.randn(batch_size, config.meta_input_dim).to(device)
    meta_input_tensor_nan[0, 0] = float('nan')
    reward_stats_tensor = torch.randn(batch_size, 2).to(device)
    lr_actor, lr_critic, lr_alpha, tau, gamma = controller(meta_input_tensor_nan, reward_stats_tensor)
    assert not torch.isnan(lr_actor).all()

    # Test with Inf values in reward_stats
    meta_input_tensor = torch.randn(batch_size, config.meta_input_dim).to(device)
    reward_stats_tensor_inf = torch.randn(batch_size, 2).to(device)
    reward_stats_tensor_inf[0, 0] = float('inf')
    lr_actor, lr_critic, lr_alpha, tau, gamma = controller(meta_input_tensor, reward_stats_tensor_inf)
    assert not torch.isinf(lr_actor).all()

    # Test with empty tensor
    meta_input_tensor_empty = torch.empty(0, config.meta_input_dim).to(device)
    reward_stats_tensor_empty = torch.empty(0, 2).to(device)
    lr_actor, lr_critic, lr_alpha, tau, gamma = controller(meta_input_tensor_empty, reward_stats_tensor_empty)
    assert lr_actor.shape == (0,)

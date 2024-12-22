# tests/test_sac.py
import pytest
import torch
from networks import MetaSACActor, MetaSACCritic
from dataclasses import dataclass

@dataclass
class Config:
    state_dim: int = 4
    action_dim: int = 2
    hidden_dim: int = 32
    num_mlp_layers: int = 3
    dropout_rate: float = 0.1
    time_encoding_dim: int = 10
    attention_dim: int = 16
    window_size: int = 10
    custom_layers: list = None
    device: torch.device = torch.device("cpu")  # CHANGED: Added device attribute

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_meta_sac_actor(device):
    config = Config()
    batch_size = 5
    actor = MetaSACActor(config).to(device)
    state_tensor = torch.randn(batch_size, config.state_dim).to(device)
    time_tensor = torch.arange(batch_size).to(device)
    mu, log_sigma = actor(state_tensor, time_tensor)
    assert mu.shape == (batch_size, config.action_dim)
    assert log_sigma.shape == (batch_size, config.action_dim)
    assert not torch.isnan(mu).any()
    assert not torch.isnan(log_sigma).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_meta_sac_critic(device):
    config = Config()
    batch_size = 5
    critic = MetaSACCritic(config).to(device)
    state_tensor = torch.randn(batch_size, config.state_dim).to(device)
    action_tensor = torch.randn(batch_size, config.action_dim).to(device)
    time_tensor = torch.arange(batch_size).to(device)
    q_value = critic(state_tensor, action_tensor, time_tensor)
    assert q_value.shape == (batch_size, 1)
    assert not torch.isnan(q_value).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_meta_sac_actor_edge_cases(device):
    config = Config()
    batch_size = 5
    actor = MetaSACActor(config).to(device)

    # Test with NaN values
    state_tensor_nan = torch.randn(batch_size, config.state_dim).to(device)
    state_tensor_nan[0, 0] = float('nan')
    time_tensor = torch.arange(batch_size).to(device)
    mu, log_sigma = actor(state_tensor_nan, time_tensor)
    assert not torch.isnan(mu).all()

    # Test with Inf values
    state_tensor_inf = torch.randn(batch_size, config.state_dim).to(device)
    state_tensor_inf[0, 0] = float('inf')
    mu, log_sigma = actor(state_tensor_inf, time_tensor)
    assert not torch.isinf(mu).all()

    # Test with empty tensor
    state_tensor_empty = torch.empty(0, config.state_dim).to(device)
    time_tensor_empty = torch.empty(0, dtype=torch.int64).to(device)
    mu, log_sigma = actor(state_tensor_empty, time_tensor_empty)
    assert mu.shape == (0, config.action_dim)

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_meta_sac_critic_edge_cases(device):
    config = Config()
    batch_size = 5
    critic = MetaSACCritic(config).to(device)

    # Test with NaN values in state
    state_tensor_nan = torch.randn(batch_size, config.state_dim).to(device)
    state_tensor_nan[0, 0] = float('nan')
    action_tensor = torch.randn(batch_size, config.action_dim).to(device)
    time_tensor = torch.arange(batch_size).to(device)
    q_value = critic(state_tensor_nan, action_tensor, time_tensor)
    assert not torch.isnan(q_value).all()

    # Test with Inf values in action
    state_tensor = torch.randn(batch_size, config.state_dim).to(device)
    action_tensor_inf = torch.randn(batch_size, config.action_dim).to(device)
    action_tensor_inf[0, 0] = float('inf')
    q_value = critic(state_tensor, action_tensor_inf, time_tensor)
    assert not torch.isinf(q_value).all()

    # Test with empty tensor
    state_tensor_empty = torch.empty(0, config.state_dim).to(device)
    action_tensor_empty = torch.empty(0, config.action_dim).to(device)
    time_tensor_empty = torch.empty(0, dtype=torch.int64).to(device)
    q_value = critic(state_tensor_empty, action_tensor_empty, time_tensor_empty)
    assert q_value.shape == (0, 1)

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_meta_sac_custom_layers(device):
    class CustomLinear(torch.nn.Module):
        def __init__(self, hidden_dim, **kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(4, hidden_dim)

        def forward(self, x):
            return self.linear(x)

    @dataclass
    class CustomConfig(Config):
        custom_layers = [CustomLinear]
        device: torch.device = torch.device("cpu")  # CHANGED: Added device attribute

    config = CustomConfig()
    batch_size = 5

    actor = MetaSACActor(config).to(device)
    state_tensor = torch.randn(batch_size, config.state_dim).to(device)
    time_tensor = torch.arange(batch_size).to(device)
    mu, log_sigma = actor(state_tensor, time_tensor)
    assert mu.shape == (batch_size, config.action_dim)
    assert log_sigma.shape == (batch_size, config.action_dim)

    critic = MetaSACCritic(config).to(device)
    action_tensor = torch.randn(batch_size, config.action_dim).to(device)
    q_value = critic(state_tensor, action_tensor, time_tensor)
    assert q_value.shape == (batch_size, 1)

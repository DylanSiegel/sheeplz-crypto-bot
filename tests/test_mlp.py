import pytest
import torch
from networks import ModernMLP, AdaptiveModulationMLP

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
@pytest.mark.parametrize("input_dim, hidden_dim, output_dim, num_layers, use_custom_layers", [
    (4, 32, 10, 3, False),
    (4, 32, 10, 3, True),
])
def test_modern_mlp(device, input_dim, hidden_dim, output_dim, num_layers, use_custom_layers):
    batch_size = 5
    model = ModernMLP(input_dim, hidden_dim, output_dim, num_layers, use_custom_layers=use_custom_layers).to(device)
    input_tensor = torch.randn(batch_size, input_dim).to(device)
    # If custom layers are used, they expect 4 features
    # Already ensured input_dim=4
    output = model(input_tensor)
    assert output.shape == (batch_size, output_dim)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
@pytest.mark.parametrize("input_dim, hidden_dim, output_dim, num_layers, time_encoding_dim, use_custom_layers", [
    (4, 32, 10, 3, 10, False),
    (4, 32, 10, 3, 10, True),
])
def test_adaptive_modulation_mlp(device, input_dim, hidden_dim, output_dim, num_layers, time_encoding_dim, use_custom_layers):
    batch_size = 5
    model = AdaptiveModulationMLP(input_dim, hidden_dim, output_dim, num_layers, time_encoding_dim=time_encoding_dim, use_custom_layers=use_custom_layers).to(device)
    input_tensor = torch.randn(batch_size, input_dim).to(device)
    time_tensor = torch.arange(batch_size).to(device)
    # If custom layers used, input_dim must be 4 which it is
    output = model(input_tensor, time_tensor)
    assert output.shape == (batch_size, output_dim)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_modern_mlp_edge_cases(device):
    input_dim = 4
    hidden_dim = 32
    output_dim = 10
    num_layers = 3
    batch_size = 5
    model = ModernMLP(input_dim, hidden_dim, output_dim, num_layers).to(device)

    # Test with NaN values
    input_tensor_nan = torch.randn(batch_size, input_dim).to(device)
    input_tensor_nan[0, 0] = float('nan')
    output_nan = model(input_tensor_nan)
    assert not torch.isnan(output_nan).all()

    # Test with Inf values
    input_tensor_inf = torch.randn(batch_size, input_dim).to(device)
    input_tensor_inf[0, 0] = float('inf')
    output_inf = model(input_tensor_inf)
    assert not torch.isinf(output_inf).all()

    # Test with empty tensor
    input_tensor_empty = torch.empty(0, input_dim).to(device)
    output_empty = model(input_tensor_empty)
    assert output_empty.shape == (0, output_dim)

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_adaptive_modulation_mlp_edge_cases(device):
    input_dim = 4
    hidden_dim = 32
    output_dim = 10
    num_layers = 3
    time_encoding_dim = 10
    batch_size = 5
    model = AdaptiveModulationMLP(input_dim, hidden_dim, output_dim, num_layers, time_encoding_dim=time_encoding_dim).to(device)

    # Test with NaN values
    input_tensor_nan = torch.randn(batch_size, input_dim).to(device)
    input_tensor_nan[0, 0] = float('nan')
    time_tensor = torch.arange(batch_size).to(device)
    output_nan = model(input_tensor_nan, time_tensor)
    assert not torch.isnan(output_nan).all()

    # Test with Inf values
    input_tensor_inf = torch.randn(batch_size, input_dim).to(device)
    input_tensor_inf[0, 0] = float('inf')
    output_inf = model(input_tensor_inf, time_tensor)
    assert not torch.isinf(output_inf).all()

    # Test with empty tensor
    input_tensor_empty = torch.empty(0, input_dim).to(device)
    time_tensor_empty = torch.empty(0, dtype=torch.int64).to(device)
    output_empty = model(input_tensor_empty, time_tensor_empty)
    assert output_empty.shape == (0, output_dim)

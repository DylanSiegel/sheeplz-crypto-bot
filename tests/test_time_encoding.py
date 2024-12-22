# tests/test_time_encoding.py
import pytest
import torch
from networks import SinusoidalTimeEncoding

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
@pytest.mark.parametrize("time_encoding_dim, time_step", [
    (10, torch.tensor([0, 1, 2])),
    (15, torch.tensor([5, 10, 15])),
    (5, torch.tensor([10, 20, 30]))
])
def test_sinusoidal_time_encoding(device, time_encoding_dim, time_step):
    encoding = SinusoidalTimeEncoding(time_encoding_dim).to(device)
    time_step = time_step.to(device)
    output = encoding(time_step)
    assert output.shape == (time_step.shape[0], time_encoding_dim)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_sinusoidal_time_encoding_edge_cases(device):
    time_encoding_dim = 10
    encoding = SinusoidalTimeEncoding(time_encoding_dim).to(device)

    # Test with an empty tensor
    time_step_empty = torch.empty(0, dtype=torch.int64).to(device)
    output_empty = encoding(time_step_empty)
    assert output_empty.shape == (0, time_encoding_dim)

    # Test with a single large time step
    time_step_large = torch.tensor([100000], dtype=torch.int64).to(device)
    output_large = encoding(time_step_large)
    assert output_large.shape == (1, time_encoding_dim)

    # Test with very small time step
    time_step_small = torch.tensor([0.0001], dtype=torch.float32).to(device)
    output_small = encoding(time_step_small)
    assert output_small.shape == (1, time_encoding_dim)
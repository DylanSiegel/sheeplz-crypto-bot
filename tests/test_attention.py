# tests/test_attention.py
import pytest
import torch
from networks import Attention

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
@pytest.mark.parametrize("input_dim, attention_dim", [
    (16, 32),
    (32, 64),
    (64, 128)
])
def test_attention(device, input_dim, attention_dim):
    batch_size = 5
    seq_len = 10
    attention = Attention(input_dim, attention_dim).to(device)
    input_tensor = torch.randn(batch_size, seq_len, input_dim).to(device)
    output = attention(input_tensor)
    assert output.shape == (batch_size, seq_len, input_dim)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_attention_edge_cases(device):
    input_dim = 16
    attention_dim = 32
    batch_size = 5
    seq_len = 10
    attention = Attention(input_dim, attention_dim).to(device)

    # Test with NaN values
    input_tensor_nan = torch.randn(batch_size, seq_len, input_dim).to(device)
    input_tensor_nan[0, 0, 0] = float('nan')
    output_nan = attention(input_tensor_nan)
    assert output_nan.shape == (batch_size, seq_len, input_dim)
    assert not torch.isnan(output_nan).all()

    # Test with Inf values
    input_tensor_inf = torch.randn(batch_size, seq_len, input_dim).to(device)
    input_tensor_inf[0, 0, 0] = float('inf')
    output_inf = attention(input_tensor_inf)
    assert output_inf.shape == (batch_size, seq_len, input_dim)
    assert not torch.isinf(output_inf).all()

    # Test with empty tensor
    input_tensor_empty = torch.empty(0, seq_len, input_dim).to(device)
    output_empty = attention(input_tensor_empty)
    assert output_empty.shape == (0, seq_len, input_dim)
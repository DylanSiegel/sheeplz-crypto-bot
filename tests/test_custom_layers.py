import pytest
import torch
from networks import KLinePatternLayer, VolatilityTrackingLayer, TimeWarpLayer, ExponentialMovingAverageLayer, FractalDimensionLayer

@pytest.fixture
def sample_kline_data():
    return torch.tensor([
        [10, 12, 9, 11],
        [11, 13, 10, 12],
        [12, 11, 8, 9],
        [9, 10, 8, 9],
        [10, 11, 9, 10]
        ], dtype=torch.float32)

@pytest.fixture
def sample_kline_data_nan_inf(sample_kline_data):
    data = sample_kline_data.clone()
    data[0, 0] = float('nan')
    data[1, 1] = float('inf')
    data[2, 2] = float('-inf')
    return data

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
@pytest.mark.parametrize("hidden_dim", [16, 32])
def test_kline_pattern_layer(device, sample_kline_data, hidden_dim):
    layer = KLinePatternLayer(hidden_dim).to(device)
    sample_kline_data = sample_kline_data.to(device)
    output = layer(sample_kline_data)
    assert output.shape == (sample_kline_data.shape[0], hidden_dim)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_kline_pattern_layer_single_input(device):
    layer = KLinePatternLayer(16).to(device)
    single_input = torch.tensor([[10, 12, 9, 11]], dtype=torch.float32).to(device)
    output = layer(single_input)
    assert output.shape == (1, 16)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_kline_pattern_layer_invalid_input(device):
    layer = KLinePatternLayer(16).to(device)
    invalid_input = torch.rand(5, 3).to(device)
    with pytest.raises(ValueError, match="Input tensor must have 4 features: open, high, low, close"):
        layer(invalid_input)
    
    invalid_input_type = [1, 2, 3, 4]
    with pytest.raises(TypeError, match="Input must be a torch.Tensor"):
        layer(invalid_input_type)

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_kline_pattern_layer_nan_inf(device, sample_kline_data_nan_inf):
    layer = KLinePatternLayer(16).to(device)
    sample_kline_data_nan_inf = sample_kline_data_nan_inf.to(device)
    output = layer(sample_kline_data_nan_inf)
    assert output.shape[0] == sample_kline_data_nan_inf.shape[0]

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
@pytest.mark.parametrize("hidden_dim, window_size", [(16, 5), (32, 10)])
def test_volatility_tracking_layer(device, sample_kline_data, hidden_dim, window_size):
    layer = VolatilityTrackingLayer(hidden_dim, window_size).to(device)
    sample_kline_data = sample_kline_data.to(device)
    output = layer(sample_kline_data)
    assert output.shape == (sample_kline_data.shape[0], hidden_dim)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_volatility_tracking_layer_invalid_input(device):
    layer = VolatilityTrackingLayer(16, 10).to(device)
    invalid_input = torch.rand(5, 3).to(device)
    with pytest.raises(ValueError, match="Input tensor must have 4 features: open, high, low, close"):
        layer(invalid_input)

    invalid_input_type = [1, 2, 3, 4]
    with pytest.raises(TypeError, match="Input must be a torch.Tensor"):
        layer(invalid_input_type)

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_volatility_tracking_layer_nan_inf(device, sample_kline_data_nan_inf):
    layer = VolatilityTrackingLayer(16, 10).to(device)
    sample_kline_data_nan_inf = sample_kline_data_nan_inf.to(device)
    output = layer(sample_kline_data_nan_inf)
    assert output.shape[0] == sample_kline_data_nan_inf.shape[0]

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
@pytest.mark.parametrize("hidden_dim, window_size", [(16, 5), (32, 10)])
def test_time_warp_layer(device, sample_kline_data, hidden_dim, window_size):
    layer = TimeWarpLayer(hidden_dim, window_size).to(device)
    sample_kline_data = sample_kline_data.to(device)
    output = layer(sample_kline_data)
    assert output.shape == (sample_kline_data.shape[0], hidden_dim)
    assert not torch.isnan(output).any()
    output = layer(sample_kline_data)
    assert output.shape == (sample_kline_data.shape[0], hidden_dim)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_time_warp_layer_invalid_input(device):
    layer = TimeWarpLayer(16, 10).to(device)
    invalid_input = torch.rand(5, 3).to(device)
    with pytest.raises(ValueError, match="Input tensor must have 4 features: open, high, low, close"):
        layer(invalid_input)

    invalid_input_type = [1, 2, 3, 4]
    with pytest.raises(TypeError, match="Input must be a torch.Tensor"):
        layer(invalid_input_type)

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_time_warp_layer_nan_inf(device, sample_kline_data_nan_inf):
    layer = TimeWarpLayer(16, 10).to(device)
    sample_kline_data_nan_inf = sample_kline_data_nan_inf.to(device)
    output = layer(sample_kline_data_nan_inf)
    assert output.shape[0] == sample_kline_data_nan_inf.shape[0]

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
@pytest.mark.parametrize("window_size, hidden_dim", [(5, 16), (10, 32)])
def test_exponential_moving_average_layer(device, sample_kline_data, window_size, hidden_dim):
    layer = ExponentialMovingAverageLayer(window_size, hidden_dim).to(device)
    sample_kline_data = sample_kline_data.to(device)
    output = layer(sample_kline_data)
    assert output.shape == (sample_kline_data.shape[0], hidden_dim)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_exponential_moving_average_layer_invalid_input(device):
    layer = ExponentialMovingAverageLayer(10, 16).to(device)
    invalid_input = torch.rand(5, 3).to(device)
    with pytest.raises(ValueError, match="Input tensor must have 4 features: open, high, low, close"):
         layer(invalid_input)

    invalid_input_type = [1, 2, 3, 4]
    with pytest.raises(TypeError, match="Input must be a torch.Tensor"):
        layer(invalid_input_type)

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_exponential_moving_average_layer_nan_inf(device, sample_kline_data_nan_inf):
    layer = ExponentialMovingAverageLayer(10, 16).to(device)
    sample_kline_data_nan_inf = sample_kline_data_nan_inf.to(device)
    output = layer(sample_kline_data_nan_inf)
    assert output.shape[0] == sample_kline_data_nan_inf.shape[0]

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
@pytest.mark.parametrize("hidden_dim, max_k", [(16, 5), (32, 10)])
def test_fractal_dimension_layer(device, sample_kline_data, hidden_dim, max_k):
    layer = FractalDimensionLayer(hidden_dim, max_k).to(device)
    sample_kline_data = sample_kline_data.to(device)
    output = layer(sample_kline_data)
    assert output.shape == (sample_kline_data.shape[0], hidden_dim)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_fractal_dimension_layer_invalid_input(device):
    layer = FractalDimensionLayer(16).to(device)
    invalid_input = torch.rand(5, 3).to(device)
    with pytest.raises(ValueError, match="Input tensor must have 4 features: open, high, low, close"):
         layer(invalid_input)

    invalid_input_type = [1, 2, 3, 4]
    with pytest.raises(TypeError, match="Input must be a torch.Tensor"):
        layer(invalid_input_type)

@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))])
def test_fractal_dimension_layer_nan_inf(device, sample_kline_data_nan_inf):
    layer = FractalDimensionLayer(16, 10).to(device)
    sample_kline_data_nan_inf = sample_kline_data_nan_inf.to(device)
    output = layer(sample_kline_data_nan_inf)
    assert output.shape[0] == sample_kline_data_nan_inf.shape[0]

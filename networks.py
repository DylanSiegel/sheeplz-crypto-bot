# networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class APELU(nn.Module):
    """Advanced Parametric Exponential Linear Unit activation."""
    def __init__(self, alpha_init: float = 0.01, beta_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle nan values first by propagating them
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            out = torch.where(nan_mask, x, torch.tensor(0.0, device=x.device, dtype=x.dtype))  # Initialize with 0s where not nan
        else:
            out = torch.zeros_like(x)

        # Apply APELU logic where x is not nan
        non_nan_mask = ~nan_mask
        out = torch.where(non_nan_mask & (x >= 0), x, out)
        out = torch.where(non_nan_mask & (x < 0), self.alpha * x * torch.exp(self.beta * x), out)

        # For inf values, ensure we don't have inf * 0
        inf_mask = torch.isinf(x)
        if inf_mask.any():
            out = torch.where(inf_mask, x, out)

        return out

class MomentumActivation(nn.Module):
    """Activation function sensitive to price momentum."""
    def __init__(self, momentum_sensitivity: float = 1.0):
        super().__init__()
        self.momentum_sensitivity = nn.Parameter(torch.tensor(momentum_sensitivity, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle nan values by propagating them
        nan_mask = torch.isnan(x)
        out = torch.where(nan_mask, x, torch.zeros_like(x))

        # Apply momentum activation logic only where x is not NaN
        non_nan_mask = ~nan_mask
        out = torch.where(non_nan_mask, x * (1 + self.momentum_sensitivity * torch.tanh(x)), out)
        
        #Handle inf values
        inf_mask = torch.isinf(x)
        if inf_mask.any():
             out = torch.where(inf_mask, x, out)
        return out

class VolatilityAdaptiveActivation(nn.Module):
    """Activation function that adapts based on volatility."""
    def __init__(self, initial_scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        # Handle nan in x by propagating them
        nan_mask_x = torch.isnan(x)
        out = torch.where(nan_mask_x, x, torch.tensor(0.0, device=x.device, dtype=x.dtype))

        # Handle nan in volatility (treat as 0)
        volatility = torch.where(torch.isnan(volatility), torch.tensor(0.0, device=volatility.device, dtype=volatility.dtype), volatility)

        # Apply activation logic only where x is not nan
        non_nan_mask_x = ~nan_mask_x
        out = torch.where(non_nan_mask_x, x * (1 + self.scale * torch.tanh(volatility)), out)

        return out

class KLinePatternLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(3, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != 4:
            raise ValueError("Input tensor must have 4 features: open, high, low, close")

        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device, dtype=x.dtype), x)

        patterns = self.detect_patterns(x)
        patterns = torch.where(torch.isnan(patterns), torch.tensor(0.0, device=patterns.device, dtype=patterns.dtype), patterns)
        return F.relu(self.linear(patterns))

    def detect_patterns(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, features = x.shape

        if batch_size == 0:
            return torch.zeros((0,3), device=x.device)

        open_prices = x[:, 0]
        close_prices = x[:, 3]
        patterns = torch.zeros((batch_size, 3), device=x.device)

        if batch_size > 1:
            prev_open = open_prices[:-1]
            prev_close = close_prices[:-1]
            curr_open = open_prices[1:]
            curr_close = close_prices[1:]

            bullish_engulfing = (prev_close < prev_open) & (curr_close > curr_open) & (curr_open < prev_close) & (curr_close > prev_open)
            bearish_engulfing = (prev_close > prev_open) & (curr_close < curr_open) & (curr_open > prev_close) & (curr_close < prev_open)
            no_pattern = ~(bullish_engulfing | bearish_engulfing)

            patterns[1:, 0] = bullish_engulfing.float()
            patterns[1:, 1] = bearish_engulfing.float()
            patterns[1:, 2] = no_pattern.float()

        # The first row has no previous candle to compare, default to no pattern
        patterns[0, 2] = 1
        return patterns

class VolatilityTrackingLayer(nn.Module):
    def __init__(self, hidden_dim: int, window_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.linear = nn.Linear(1, hidden_dim)
        self.close_prices_buffer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != 4:
            raise ValueError("Input tensor must have 4 features: open, high, low, close")

        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device, dtype=x.dtype), x)

        close_prices = x[:, 3].unsqueeze(-1)
        self.update_buffer(close_prices)

        if self.close_prices_buffer.shape[1] >= self.window_size:
            volatility_vectors = self.calculate_volatility_vectors()
        else:
            volatility_vectors = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

        volatility_vectors = torch.where(torch.isnan(volatility_vectors), torch.tensor(0.0, device=volatility_vectors.device, dtype=volatility_vectors.dtype), volatility_vectors)
        return F.relu(self.linear(volatility_vectors))

    def update_buffer(self, close_prices: torch.Tensor):
        if self.close_prices_buffer is None:
            self.close_prices_buffer = close_prices
        else:
            buffer_len = self.close_prices_buffer.shape[1]
            if buffer_len < self.window_size:
                self.close_prices_buffer = torch.cat([self.close_prices_buffer, close_prices], dim=1)
            else:
                self.close_prices_buffer = torch.cat([self.close_prices_buffer[:, 1:], close_prices], dim=1)

    def calculate_volatility_vectors(self) -> torch.Tensor:
        log_returns = torch.log(self.close_prices_buffer[:, 1:] / self.close_prices_buffer[:, :-1])
        log_returns = torch.where(torch.isnan(log_returns), torch.tensor(0.0, device=log_returns.device, dtype=log_returns.dtype), log_returns)
        log_returns = torch.where(torch.isinf(log_returns), torch.tensor(1.0, device=log_returns.device, dtype=log_returns.dtype), log_returns)
        volatility = torch.std(log_returns, dim=1, unbiased=True).unsqueeze(-1)
        return volatility

class TimeWarpLayer(nn.Module):
    def __init__(self, hidden_dim: int, window_size: int = 10):
        super().__init__()
        self.linear = nn.Linear(4, hidden_dim)
        self.window_size = window_size
        self.last_x = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != 4:
            raise ValueError("Input tensor must have 4 features: open, high, low, close")

        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device, dtype=x.dtype), x)

        if self.last_x is None:
            self.last_x = x

        time_warped_x = self.time_warp(self.last_x, x)
        self.last_x = x.clone().detach()

        time_warped_x = torch.where(torch.isnan(time_warped_x), torch.tensor(0.0, device=x.device, dtype=x.dtype), time_warped_x)
        return self.linear(time_warped_x)

    def time_warp(self, last_x: torch.Tensor, current_x: torch.Tensor) -> torch.Tensor:
        warped_x = (last_x + current_x) / 2
        return warped_x

class ExponentialMovingAverageLayer(nn.Module):
    def __init__(self, window_size: int, hidden_dim: int):
        super().__init__()
        self.window_size = window_size
        self.alpha = 2 / (window_size + 1)
        self.ema = None
        self.linear = nn.Linear(1, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != 4:
            raise ValueError("Input tensor must have 4 features: open, high, low, close")

        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device, dtype=x.dtype), x)
        close_prices = x[:, 3]

        if self.ema is None:
            self.ema = close_prices.clone()
        else:
            self.ema = (close_prices * self.alpha) + (self.ema * (1 - self.alpha))

        ema_values = self.ema.unsqueeze(-1)
        ema_values = torch.where(torch.isnan(ema_values), torch.tensor(0.0, device=x.device, dtype=ema_values.dtype), ema_values)
        return F.relu(self.linear(ema_values))

class FractalDimensionLayer(nn.Module):
    def __init__(self, hidden_dim: int, max_k: int = 10, buffer_size: int = 50):
        super().__init__()
        self.linear = nn.Linear(1, hidden_dim)
        self.max_k = max_k
        self.buffer_size = buffer_size
        self.values_buffer = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != 4:
            raise ValueError("Input tensor must have 4 features: open, high, low, close")

        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device, dtype=x.dtype), x)

        close_prices = x[:, 3]
        hfd_values = []

        for price in close_prices:
            self.update_buffer(price.item())
            if len(self.values_buffer) > self.max_k:
                hfd = self.calculate_hfd()
                hfd_values.append(hfd)
            else:
                hfd_values.append(0.0)

        hfd_tensor = torch.tensor(hfd_values, device=x.device, dtype=torch.float32).unsqueeze(-1)
        hfd_tensor = torch.where(torch.isnan(hfd_tensor), torch.tensor(0.0, device=x.device, dtype=hfd_tensor.dtype), hfd_tensor)
        return F.relu(self.linear(hfd_tensor))

    def update_buffer(self, value: float):
        self.values_buffer.append(value)
        if len(self.values_buffer) > self.buffer_size:
            self.values_buffer.pop(0)

    def calculate_hfd(self):
        if len(self.values_buffer) < self.max_k + 1:
            return 0.0

        arr = np.array(self.values_buffer)
        lk_values = []
        for k in range(1, self.max_k + 1):
            lk_total = 0
            for m in range(k):
                indexes = np.arange(m, len(arr), k)
                if len(indexes) >= 2:
                    lengths = np.abs(np.diff(arr[indexes]))
                    lk_total += np.sum(lengths) * (len(arr) - 1) / (len(indexes)*k)

            if k > 0 and lk_total > 0:
                lk = lk_total / k
                lk_values.append(lk)

        if len(lk_values) > 1:
            k_values = np.arange(1, len(lk_values) + 1)
            log_k = np.log(k_values)
            log_lk = np.log(lk_values)

            log_k = np.nan_to_num(log_k, nan=0.0, posinf=1.0, neginf=-1.0)
            log_lk = np.nan_to_num(log_lk, nan=0.0, posinf=1.0, neginf=-1.0)

            slope, _ = np.polyfit(log_k, log_lk, 1)
            return -slope
        else:
            return 0.0

class ModernMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout_rate: float = 0.1, use_custom_layers: bool = False, window_size:int = 10, custom_layers: list = None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_custom_layers = use_custom_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.custom_layers_list = []
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_custom_layers:
            if custom_layers is None:
                self.custom_layers_list = [
                    KLinePatternLayer(hidden_dim),
                    VolatilityTrackingLayer(hidden_dim, window_size),
                    TimeWarpLayer(hidden_dim, window_size),
                    ExponentialMovingAverageLayer(window_size, hidden_dim),
                    FractalDimensionLayer(hidden_dim)
                ]
            else:
                self.custom_layers_list = [layer(hidden_dim=hidden_dim, window_size=window_size) for layer in custom_layers]

            for i, cl in enumerate(self.custom_layers_list):
                cl.to(self.device_)

            in_features = hidden_dim * len(self.custom_layers_list)
        else:
            in_features = input_dim

        for i in range(num_layers):
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            layer_ = nn.Linear(in_features if i == 0 else hidden_dim, out_features)
            layer_.to(self.device_)
            self.layers.append(layer_)
            if i != num_layers - 1:
                norm_ = nn.LayerNorm(hidden_dim).to(self.device_)
                self.norms.append(norm_)
        self.activation = APELU().to(self.device_)
        self.dropout = nn.Dropout(dropout_rate).to(self.device_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_custom_layers:
            custom_outputs = []
            for layer in self.custom_layers_list:
                layer = layer.to(x.device)
                out = layer(x)
                custom_outputs.append(out)
            x = torch.cat(custom_outputs, dim=-1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        return x

class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, time_encoding_dim: int):
        super().__init__()
        self.time_encoding_dim = time_encoding_dim
        self.frequencies = 10 ** (torch.arange(0, time_encoding_dim // 2) * (-2 / (time_encoding_dim // 2)))

    def forward(self, time_step: torch.Tensor) -> torch.Tensor:
        time_step = time_step.float()
        scaled_time = time_step.unsqueeze(-1) * self.frequencies.to(time_step.device)
        sin_encodings = torch.sin(scaled_time)
        cos_encodings = torch.cos(scaled_time)

        if self.time_encoding_dim % 2 == 0:
            encoding = torch.cat([sin_encodings, cos_encodings], dim=-1)
        else:
            encoding = torch.cat([sin_encodings, cos_encodings, torch.zeros_like(cos_encodings[:, :1])], dim=-1)
        return encoding

class TimeAwareBias(nn.Module):
    def __init__(self, input_dim: int, time_encoding_dim: int = 10, hidden_dim: int = 20):
        super().__init__()
        self.time_embedding = nn.Linear(time_encoding_dim, hidden_dim)
        self.time_projection = nn.Linear(hidden_dim, input_dim)
        self.activation = APELU()

    def forward(self, time_encoding: torch.Tensor) -> torch.Tensor:
        x = self.time_embedding(time_encoding)
        x = self.activation(x)
        return self.time_projection(x)

class AdaptiveModulationMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout_rate: float = 0.1, time_encoding_dim: int = 10, use_custom_layers: bool = False, window_size:int = 10, custom_layers: list = None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.modulations = nn.ParameterList()
        self.time_biases = nn.ModuleList()
        self.sinusoidal_encoding = SinusoidalTimeEncoding(time_encoding_dim)
        self.use_custom_layers = use_custom_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.custom_layers_list = []
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_custom_layers:
            if custom_layers is None:
                self.custom_layers_list = [
                    KLinePatternLayer(hidden_dim),
                    VolatilityTrackingLayer(hidden_dim, window_size),
                    TimeWarpLayer(hidden_dim, window_size),
                    ExponentialMovingAverageLayer(window_size, hidden_dim),
                    FractalDimensionLayer(hidden_dim)
                ]
            else:
                self.custom_layers_list = [layer(hidden_dim=hidden_dim, window_size=window_size) for layer in custom_layers]

            for cl in self.custom_layers_list:
                cl.to(self.device_)

            in_features = hidden_dim * len(self.custom_layers_list)
        else:
            in_features = input_dim

        for i in range(num_layers):
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            layer_ = nn.Linear(in_features if i == 0 else hidden_dim, out_features).to(self.device_)
            self.layers.append(layer_)
            if i != num_layers - 1:
                norm_ = nn.LayerNorm(hidden_dim).to(self.device_)
                self.norms.append(norm_)
                self.modulations.append(nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32)))
                self.time_biases.append(TimeAwareBias(hidden_dim, time_encoding_dim).to(self.device_))

        self.activation = APELU().to(self.device_)
        self.dropout = nn.Dropout(dropout_rate).to(self.device_)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        time_encoding = self.sinusoidal_encoding(time_step).to(x.device)

        if self.use_custom_layers:
            custom_outputs = []
            for layer in self.custom_layers_list:
                layer = layer.to(x.device)
                out = layer(x)
                custom_outputs.append(out)
            x = torch.cat(custom_outputs, dim=-1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.num_layers - 1:
                modulation_factor = self.modulations[i].to(x.device) + self.time_biases[i](time_encoding)
                x = x * modulation_factor
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)

        return x

class Attention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, attention_dim)
        self.key_proj = nn.Linear(input_dim, attention_dim)
        self.value_proj = nn.Linear(input_dim, attention_dim)
        self.out_proj = nn.Linear(attention_dim, input_dim)

        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.where(torch.isnan(input), torch.tensor(0.0, device=input.device, dtype=input.dtype), input)

        query = self.query_proj(input)
        key = self.key_proj(input)
        value = self.value_proj(input)

        attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        return self.out_proj(attn_output)

class MetaSACActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config.state_dim, config.attention_dim).to(config.device)
        self.mlp = AdaptiveModulationMLP(
            config.state_dim,
            config.hidden_dim,
            2 * config.action_dim,
            config.num_mlp_layers,
            config.dropout_rate,
            config.time_encoding_dim,
            use_custom_layers=(config.custom_layers is not None),
            window_size=config.window_size,
            custom_layers=config.custom_layers
        ).to(config.device)
        self.action_dim = config.action_dim

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.to(next(self.parameters()).device)
        time_step = time_step.to(x.device)
        x = self.attention(x)
        x = x.squeeze(1)
        x = self.mlp(x, time_step)

        mu, log_sigma = x[:, :self.action_dim], x[:, self.action_dim:]
        mu = torch.where(torch.isnan(mu), torch.tensor(0.0, device=mu.device, dtype=mu.dtype), mu)
        log_sigma = torch.where(torch.isnan(log_sigma), torch.tensor(0.0, device=log_sigma.device, dtype=log_sigma.dtype), log_sigma)
        return torch.tanh(mu), log_sigma

class MetaSACCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        combined_dim = config.state_dim + config.action_dim
        self.attention = Attention(combined_dim, config.attention_dim).to(config.device)
        self.mlp = AdaptiveModulationMLP(
            combined_dim,
            config.hidden_dim,
            1,
            config.num_mlp_layers,
            config.dropout_rate,
            config.time_encoding_dim,
            use_custom_layers=(config.custom_layers is not None),
            window_size=config.window_size,
            custom_layers=config.custom_layers
        ).to(config.device)

    def forward(self, state: torch.Tensor, action: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = x.to(next(self.parameters()).device)
        time_step = time_step.to(x.device)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.attention(x)
        x = x.squeeze(1)
        x = self.mlp(x, time_step)
        return x

class MetaController(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = ModernMLP(
            config.meta_input_dim + 2,
            config.hidden_dim,
            config.num_hyperparams,
            config.num_mlp_layers,
            config.dropout_rate,
            use_custom_layers=False,
            window_size=config.window_size
        ).to(config.device)
        self.num_hyperparams = config.num_hyperparams

    def forward(self, x: torch.Tensor, reward_stats: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = x.to(next(self.parameters()).device)
        reward_stats = reward_stats.to(x.device)
        x = torch.cat([x, reward_stats], dim=-1)
        hyperparameter_outputs = self.mlp(x)

        learning_rate_actor = torch.sigmoid(hyperparameter_outputs[:, 0])
        learning_rate_critic = torch.sigmoid(hyperparameter_outputs[:, 1])
        learning_rate_alpha = torch.sigmoid(hyperparameter_outputs[:, 2])
        tau = torch.sigmoid(hyperparameter_outputs[:, 3])
        gamma = 0.9 + 0.09 * torch.sigmoid(hyperparameter_outputs[:, 4])

        return learning_rate_actor, learning_rate_critic, learning_rate_alpha, tau, gamma
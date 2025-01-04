import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from config import EnvironmentConfig

class APELU(nn.Module):
    def __init__(self, alpha_init: float = 0.01, beta_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.where(x >= 0, x, self.alpha * x * torch.exp(self.beta * x))
        out = torch.nan_to_num(out)
        return out

class MomentumActivation(nn.Module):
    def __init__(self, momentum_sensitivity: float = 1.0):
        super().__init__()
        self.momentum_sensitivity = nn.Parameter(torch.tensor(momentum_sensitivity, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * (1 + self.momentum_sensitivity * torch.tanh(x))
        out = torch.nan_to_num(out)
        return out

class VolatilityAdaptiveActivation(nn.Module):
    def __init__(self, initial_scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        volatility = torch.nan_to_num(volatility, nan=0.0)
        out = x * (1 + self.scale * torch.tanh(volatility))
        out = torch.nan_to_num(out)
        return out

class KLinePatternLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(5, hidden_dim)  # Bullish, Bearish, Doji, Hammer, Inverted Hammer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patterns = self.detect_patterns(x)
        patterns = torch.nan_to_num(patterns)
        return F.relu(self.linear(patterns))

    def detect_patterns(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _ = x.shape
        open_prices = x[:, 0]
        high_prices = x[:, 1]
        low_prices = x[:, 2]
        close_prices = x[:, 3]
        patterns = torch.zeros((batch_size, 5), device=x.device)

        if batch_size > 1:
            # Bullish Engulfing: Current close > current open, previous close < previous open,
            # current open < previous close, current close > previous open
            bullish = (close_prices[1:] > open_prices[1:]) & (close_prices[:-1] < open_prices[:-1]) & \
                      (open_prices[1:] < close_prices[:-1]) & (close_prices[1:] > open_prices[:-1])

            # Bearish Engulfing: Current close < current open, previous close > previous open,
            # current open > previous close, current close < previous open
            bearish = (close_prices[1:] < open_prices[1:]) & (close_prices[:-1] > open_prices[:-1]) & \
                      (open_prices[1:] > close_prices[:-1]) & (close_prices[1:] < open_prices[:-1])

            # Doji: Open and close prices are very close
            doji = torch.abs(open_prices[1:] - close_prices[1:]) < (0.05 * (high_prices[1:] - low_prices[1:]))

            # Hammer: Small body, long lower shadow, little to no upper shadow, appears at the bottom of a downtrend
            hammer = (high_prices[1:] - torch.max(open_prices[1:], close_prices[1:])) < (0.1 * (high_prices[1:] - low_prices[1:])) & \
                     (torch.min(open_prices[1:], close_prices[1:]) - low_prices[1:]) > (0.7 * (high_prices[1:] - low_prices[1:]))

            # Inverted Hammer: Small body, long upper shadow, little to no lower shadow, appears at the bottom of a downtrend
            inv_hammer = (high_prices[1:] - torch.max(open_prices[1:], close_prices[1:])) > (0.7 * (high_prices[1:] - low_prices[1:])) & \
                         (torch.min(open_prices[1:], close_prices[1:]) - low_prices[1:]) < (0.1 * (high_prices[1:] - low_prices[1:]))

            patterns[1:, 0] = bullish.float()
            patterns[1:, 1] = bearish.float()
            patterns[1:, 2] = doji.float()
            patterns[1:, 3] = hammer.float()
            patterns[1:, 4] = inv_hammer.float()

        patterns[0, :5] = 0 # No pattern initially
        return patterns

class VolatilityTrackingLayer(nn.Module):
    def __init__(self, hidden_dim: int, window_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.linear = nn.Linear(3, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        close_prices = x[:, 3].unsqueeze(-1)
        volatility_measures = self.calculate_volatility_measures(close_prices, x)
        volatility_measures = torch.nan_to_num(volatility_measures)
        return F.relu(self.linear(volatility_measures))

    def calculate_volatility_measures(self, close_prices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        window = min(self.window_size, close_prices.shape[1])
        if window < 2:
            return torch.zeros((close_prices.shape[0], 3), device=close_prices.device)

        std_dev = torch.zeros((close_prices.shape[0],), device=close_prices.device)
        garman_klass = torch.zeros((close_prices.shape[0],), device=close_prices.device)
        parkinson = torch.zeros((close_prices.shape[0],), device=close_prices.device)

        for i in range(close_prices.shape[0]):
            if close_prices.shape[1] >= window:
                window_data = close_prices[i, -window:]
                log_returns = torch.log(window_data[1:] / window_data[:-1] + 1e-8)
                std_dev[i] = torch.std(log_returns)

                # Assuming high and low prices are available as x[:, 1] and x[:, 2]
                high = x[i, 1, -window:]
                low = x[i, 2, -window:]

                log_hl = torch.log(high / low + 1e-8)
                log_cc = torch.log(window_data[1:] / window_data[:-1] + 1e-8)
                garman_klass[i] = torch.sqrt(torch.mean(0.5 * log_hl**2 - (2 * torch.log(torch.tensor(2.0)) - 1) * log_cc**2))

                parkinson[i] = torch.sqrt(torch.mean(log_hl**2) / (4 * torch.log(torch.tensor(2.0))))

        return torch.stack([std_dev, garman_klass, parkinson], dim=1)

class FractalDimensionLayer(nn.Module):
    def __init__(self, hidden_dim: int, max_k: int = 10, buffer_size: int = 50):
        super().__init__()
        self.linear = nn.Linear(1, hidden_dim)
        self.max_k = max_k
        self.buffer_size = buffer_size
        self.values_buffer = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        close_prices = x[:, 3]
        hfd_values = torch.zeros((close_prices.shape[0], 1), device=x.device)

        for i, price in enumerate(close_prices):
            self.update_buffer(price.item())
            if len(self.values_buffer) > self.max_k:
                hfd_values[i, 0] = self.calculate_hfd_optimized()
        return F.relu(self.linear(hfd_values))

    def update_buffer(self, value: float):
        self.values_buffer.append(value)
        if len(self.values_buffer) > self.buffer_size:
            self.values_buffer.pop(0)

    def calculate_hfd_optimized(self) -> float:
        if len(self.values_buffer) < self.max_k + 1:
            return 0.0

        arr = np.array(self.values_buffer)
        n = len(arr)
        lk_values = np.zeros(self.max_k)

        for k in range(1, self.max_k + 1):
            for m in range(k):
                idxs = np.arange(m, n, k)
                if len(idxs) >= 2:
                    lengths = np.abs(np.diff(arr[idxs]))
                    lk_values[k - 1] += np.sum(lengths) * (n - 1) / (len(idxs) * k)
            lk_values[k - 1] /= k

        valid_k_values = (lk_values > 0)
        if np.sum(valid_k_values) > 1:
            k_arr = np.arange(1, self.max_k + 1)[valid_k_values]
            log_k = np.log(k_arr)
            log_lk = np.log(lk_values[valid_k_values])
            slope, _ = np.polyfit(log_k, log_lk, 1)
            return -slope
        else:
            return 0.0

class MarketModeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HighLevelPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MetaController(nn.Module):
    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.meta_input_dim + 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_hyperparams + 3),
            nn.Sigmoid()
        )
        self.num_hyperparams = config.num_hyperparams
        self.ema_smoothing = 0.9 # Smoothing factor

        self.ema_values = None

    def forward(self, x: torch.Tensor, reward_stats: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        cat_input = torch.cat([x, reward_stats], dim=-1)
        out = self.mlp(cat_input)

        if self.ema_values is None:
            self.ema_values = list(out.split(1, dim=-1))
        else:
            for i, o in enumerate(out.split(1, dim=-1)):
                self.ema_values[i] = self.ema_smoothing * self.ema_values[i] + (1 - self.ema_smoothing) * o

        return tuple(self.ema_values)
    
class TimeWarpLayer(nn.Module):
    """Applies a time warping effect by averaging with last input."""
    def __init__(self, hidden_dim: int, window_size: int = 10):
        super().__init__()
        self.linear = nn.Linear(4, hidden_dim)
        self.last_x = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies time warping and linear layer."""
        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device, dtype=x.dtype), x)
        if self.last_x is None:
            self.last_x = x
        time_warped_x = (self.last_x + x) / 2
        self.last_x = x.clone().detach()
        return self.linear(time_warped_x)

class ExponentialMovingAverageLayer(nn.Module):
    """Calculates and applies EMA of close prices."""
    def __init__(self, window_size: int, hidden_dim: int):
        super().__init__()
        self.window_size = window_size
        self.alpha = 2/(window_size+1)
        self.ema = None
        self.linear = nn.Linear(1, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates EMA and applies linear layer."""
        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device), x)
        close_prices = x[:, 3]
        if self.ema is None:
            self.ema = close_prices.clone()
        else:
            self.ema = (close_prices * self.alpha) + (self.ema * (1 - self.alpha))
        ema_values = self.ema.unsqueeze(-1)
        ema_values = torch.where(torch.isnan(ema_values), torch.tensor(0.0, device=x.device), ema_values)
        return F.relu(self.linear(ema_values))
    
class SinusoidalTimeEncoding(nn.Module):
    """Encodes time using sinusoidal functions."""
    def __init__(self, time_encoding_dim: int):
        super().__init__()
        self.time_encoding_dim = time_encoding_dim
        self.frequencies = 10**(torch.arange(0, time_encoding_dim//2)*(-2/(time_encoding_dim//2)))

    def forward(self, time_step: torch.Tensor) -> torch.Tensor:
        """Applies sinusoidal time encoding."""
        time_step = time_step.float().unsqueeze(-1)
        scaled_time = time_step * self.frequencies.to(time_step.device)
        sin_enc = torch.sin(scaled_time)
        cos_enc = torch.cos(scaled_time)
        if self.time_encoding_dim % 2 == 0:
            encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        else:
            zero_pad = torch.zeros_like(cos_enc[:, :1])
            encoding = torch.cat([sin_enc, cos_enc, zero_pad], dim=-1)
        return encoding

class TimeAwareBias(nn.Module):
    """Learns a bias that is a function of time encoding."""
    def __init__(self, input_dim: int, time_encoding_dim: int, hidden_dim: int):
        super().__init__()
        self.time_embedding = nn.Linear(time_encoding_dim, hidden_dim)
        self.time_projection = nn.Linear(hidden_dim, input_dim)
        self.activation = APELU()

    def forward(self, time_encoding: torch.Tensor) -> torch.Tensor:
        """Applies time-aware bias."""
        x = self.time_embedding(time_encoding)
        x = self.activation(x)
        return self.time_projection(x)

class ModernMLP(nn.Module):
    """MLP with custom layers, layer norm, dropout, and APELU."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, dropout_rate: float, use_custom_layers: bool,
                 window_size: int, custom_layers: Optional[List[str]] = None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_custom_layers = use_custom_layers
        self.hidden_dim = hidden_dim
        self.activation = APELU()
        self.dropout = nn.Dropout(dropout_rate)

        self.custom_layers_list: List[nn.Module] = []
        if use_custom_layers:
            layer_mapping = {
                "KLinePatternLayer": lambda: KLinePatternLayer(hidden_dim),
                "VolatilityTrackingLayer": lambda: VolatilityTrackingLayer(hidden_dim, window_size),
                "TimeWarpLayer": lambda: TimeWarpLayer(hidden_dim, window_size),
                "ExponentialMovingAverageLayer": lambda: ExponentialMovingAverageLayer(window_size, hidden_dim),
                "FractalDimensionLayer": lambda: FractalDimensionLayer(hidden_dim)
            }
            if custom_layers is None:
                self.custom_layers_list = [
                    KLinePatternLayer(hidden_dim),
                    VolatilityTrackingLayer(hidden_dim, window_size),
                    TimeWarpLayer(hidden_dim, window_size),
                    ExponentialMovingAverageLayer(window_size, hidden_dim),
                    FractalDimensionLayer(hidden_dim)
                ]
            else:
                for layer_name in custom_layers:
                    if layer_name in layer_mapping:
                        self.custom_layers_list.append(layer_mapping[layer_name]())
            in_features = hidden_dim * len(self.custom_layers_list)
        else:
            in_features = input_dim

        prev_features = in_features
        for i in range(num_layers):
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(prev_features, out_features))
            if i != num_layers - 1:
                self.norms.append(nn.LayerNorm(out_features))
            prev_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies MLP with optional custom layers."""
        if self.use_custom_layers:
            outputs = []
            for cl in self.custom_layers_list:
                out = cl(x)
                outputs.append(out)
            x = torch.cat(outputs, dim=-1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        return x

class AdaptiveModulationMLP(nn.Module):
    """MLP with time-aware modulation and custom layers."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, dropout_rate: float, time_encoding_dim: int,
                 use_custom_layers: bool, window_size: int, custom_layers: Optional[List[str]]=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.modulations = nn.ParameterList()
        self.time_biases = nn.ModuleList()
        self.sinusoidal_encoding = SinusoidalTimeEncoding(time_encoding_dim)
        self.use_custom_layers = use_custom_layers
        self.activation = APELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layers = num_layers

        self.custom_layers_list: List[nn.Module] = []
        if use_custom_layers:
            layer_mapping = {
                "KLinePatternLayer": lambda: KLinePatternLayer(hidden_dim),
                "VolatilityTrackingLayer": lambda: VolatilityTrackingLayer(hidden_dim, window_size),
                "TimeWarpLayer": lambda: TimeWarpLayer(hidden_dim, window_size),
                "ExponentialMovingAverageLayer": lambda: ExponentialMovingAverageLayer(window_size, hidden_dim),
                "FractalDimensionLayer": lambda: FractalDimensionLayer(hidden_dim)
            }
            if custom_layers is None:
                self.custom_layers_list = [
                    KLinePatternLayer(hidden_dim),
                    VolatilityTrackingLayer(hidden_dim, window_size),
                    TimeWarpLayer(hidden_dim, window_size),
                    ExponentialMovingAverageLayer(window_size, hidden_dim),
                    FractalDimensionLayer(hidden_dim)
                ]
            else:
                for layer_name in custom_layers:
                    if layer_name in layer_mapping:
                        self.custom_layers_list.append(layer_mapping[layer_name]())

            in_features = hidden_dim * len(self.custom_layers_list)
        else:
            in_features = input_dim

        prev_features = in_features
        for i in range(num_layers):
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(prev_features, out_features))
            if i != num_layers - 1:
                self.norms.append(nn.LayerNorm(out_features))
                self.modulations.append(nn.Parameter(torch.ones(out_features)))
                self.time_biases.append(TimeAwareBias(out_features, time_encoding_dim, out_features))
            prev_features = out_features

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """Applies MLP with time-aware modulation."""
        time_encoding = self.sinusoidal_encoding(time_step)

        if self.use_custom_layers:
            outputs = []
            for cl in self.custom_layers_list:
                out = cl(x)
                outputs.append(out)
            x = torch.cat(outputs, dim=-1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.num_layers - 1:
                mod_factor = self.modulations[i] + self.time_biases[i](time_encoding)
                x = x * mod_factor
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        return x

class Attention(nn.Module):
    """Attention mechanism."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies attention mechanism."""
        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device, dtype=x.dtype), x)
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        return self.out_proj(attn_output)

class MetaSACActor(nn.Module):
    """Actor network for MetaSAC."""
    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.attention = Attention(config.state_dim, config.attention_dim)
        self.mlp = AdaptiveModulationMLP(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=2*config.action_dim,
            num_layers=config.num_mlp_layers,
            dropout_rate=config.dropout_rate,
            time_encoding_dim=config.time_encoding_dim,
            use_custom_layers=bool(config.custom_layers),
            window_size=config.window_size,
            custom_layers=config.custom_layers
        )
        self.action_dim = config.action_dim

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Outputs mean and log sigma for action sampling."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.attention(x)
        x = x.squeeze(1)
        x = self.mlp(x, time_step)
        mu = x[:, :self.action_dim]
        log_sigma = x[:, self.action_dim:]
        mu = torch.tanh(mu)
        return mu, log_sigma

class MetaSACCritic(nn.Module):
    """Critic network for MetaSAC."""
    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        combined_dim = config.state_dim + config.action_dim
        self.attention = Attention(combined_dim, config.attention_dim)
        self.mlp = AdaptiveModulationMLP(
            input_dim=combined_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_layers=config.num_mlp_layers,
            dropout_rate=config.dropout_rate,
            time_encoding_dim=config.time_encoding_dim,
            use_custom_layers=bool(config.custom_layers),
            window_size=config.window_size,
            custom_layers=config.custom_layers
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """Outputs Q-value estimate."""
        x = torch.cat([state, action], dim=-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.attention(x)
        x = x.squeeze(1)
        x = self.mlp(x, time_step)
        return x

class PolicyDistiller(nn.Module):
    """Combines outputs from multiple specialist policies."""
    def __init__(self, specialist_policies: List[nn.Module]):
        super().__init__()
        self.specialists = nn.ModuleList(specialist_policies)

    def forward(self, state: torch.Tensor, time_step: torch.Tensor):
        """Averages outputs from specialist policies."""
        outputs = [spec(state, time_step) for spec in self.specialists]
        mus = torch.stack([o[0] for o in outputs], dim=0)
        log_sigmas = torch.stack([o[1] for o in outputs], dim=0)
        mu_avg = torch.mean(mus, dim=0)
        log_sigma_avg = torch.mean(log_sigmas, dim=0)
        return mu_avg, log_sigma_avg

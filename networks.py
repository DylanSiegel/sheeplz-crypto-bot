# File: networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from torch_geometric.nn import GCNConv
import math

from config import EnvironmentConfig

# ============================
# Custom Activation Functions and Layers
# ============================

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

# ============================
# Specialized Layers for Financial Data
# ============================

class KLinePatternLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(5, hidden_dim)  # Bullish, Bearish, Doji, Hammer, Inverted Hammer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patterns = self.detect_patterns(x)
        patterns = torch.nan_to_num(patterns)
        return F.relu(self.linear(patterns))

    def detect_patterns(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        patterns = torch.zeros((batch_size, seq_length, 5), device=x.device)

        if batch_size > 1 and seq_length > 1:
            open_prices = x[:, :-1, 0]
            high_prices = x[:, :-1, 1]
            low_prices = x[:, :-1, 2]
            close_prices = x[:, :-1, 3]
            prev_open = x[:, 1:, 0]
            prev_close = x[:, 1:, 3]

            # Bullish Engulfing
            bullish = (close_prices > open_prices) & (prev_close < prev_open) & \
                      (open_prices < prev_close) & (close_prices > prev_open)

            # Bearish Engulfing
            bearish = (close_prices < open_prices) & (prev_close > prev_open) & \
                      (open_prices > prev_close) & (close_prices < prev_open)

            # Doji
            doji = torch.abs(open_prices - close_prices) < (0.05 * (high_prices - low_prices))

            # Hammer
            hammer = ((high_prices - torch.max(open_prices, close_prices)) < (0.1 * (high_prices - low_prices))) & \
                     ((torch.min(open_prices, close_prices) - low_prices) > (0.7 * (high_prices - low_prices)))

            # Inverted Hammer
            inv_hammer = ((high_prices - torch.max(open_prices, close_prices)) > (0.7 * (high_prices - low_prices))) & \
                         ((torch.min(open_prices, close_prices) - low_prices) < (0.1 * (high_prices - low_prices)))

            patterns[:, 1:, 0] = bullish.float()
            patterns[:, 1:, 1] = bearish.float()
            patterns[:, 1:, 2] = doji.float()
            patterns[:, 1:, 3] = hammer.float()
            patterns[:, 1:, 4] = inv_hammer.float()

        return patterns

class VolatilityTrackingLayer(nn.Module):
    def __init__(self, hidden_dim: int, window_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.linear = nn.Linear(3, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        close_prices = x[:, :, 3].unsqueeze(-1)  # (batch_size, seq_length, 1)
        volatility_measures = self.calculate_volatility_measures(close_prices, x)
        volatility_measures = torch.nan_to_num(volatility_measures)
        return F.relu(self.linear(volatility_measures))

    def calculate_volatility_measures(self, close_prices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = close_prices.shape
        measures = torch.zeros((batch_size, seq_length, 3), device=close_prices.device)

        for i in range(batch_size):
            for t in range(self.window_size, seq_length):
                window = close_prices[i, t - self.window_size:t, 0]
                log_returns = torch.log(window[1:] / window[:-1] + 1e-8)
                std_dev = torch.std(log_returns)

                high = x[i, t, 1]
                low = x[i, t, 2]
                log_hl = torch.log(high / low + 1e-8)
                log_cc = log_returns
                garman_klass = torch.sqrt(torch.mean(0.5 * log_hl**2 - (2 * torch.log(torch.tensor(2.0)) - 1) * log_cc**2))
                parkinson = torch.sqrt(torch.mean(log_hl**2) / (4 * torch.log(torch.tensor(2.0))))

                measures[i, t, 0] = std_dev
                measures[i, t, 1] = garman_klass
                measures[i, t, 2] = parkinson

        return measures

class TimeWarpLayer(nn.Module):
    def __init__(self, hidden_dim: int, window_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement time warping logic
        # Placeholder implementation
        return F.relu(self.linear(x))

class ExponentialMovingAverageLayer(nn.Module):
    def __init__(self, window_size: int, hidden_dim: int):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement EMA logic
        # Placeholder implementation
        return F.relu(self.linear(x))

class FractalDimensionLayer(nn.Module):
    def __init__(self, hidden_dim: int, max_k: int = 10, buffer_size: int = 50):
        super().__init__()
        self.linear = nn.Linear(1, hidden_dim)
        self.max_k = max_k
        self.buffer_size = buffer_size
        self.register_buffer('values_buffer', torch.zeros(buffer_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        close_prices = x[:, :, 3]  # (batch_size, seq_length)
        batch_size, seq_length = close_prices.shape
        hfd_values = torch.zeros((batch_size, seq_length, 1), device=x.device)

        for i in range(batch_size):
            for t in range(seq_length):
                # Update buffer: shift left and append new price
                self.values_buffer = torch.roll(self.values_buffer, shifts=-1)
                self.values_buffer[-1] = close_prices[i, t]

                if t >= self.max_k:
                    window = self.values_buffer[-self.max_k:]
                    hfd_values[i, t, 0] = self.calculate_hfd_optimized(window)

        return F.relu(self.linear(hfd_values))

    def calculate_hfd_optimized(self, arr: torch.Tensor) -> float:
        n = len(arr)
        if n < self.max_k + 1:
            return 0.0

        lk_values = torch.zeros(self.max_k, device=arr.device)
        for k in range(1, self.max_k + 1):
            for m in range(k):
                idxs = torch.arange(m, n, k, device=arr.device)
                if len(idxs) >= 2:
                    lengths = torch.abs(arr[idxs[1:]] - arr[idxs[:-1]])
                    lk_values[k - 1] += torch.sum(lengths) * (n - 1) / (len(idxs) * k)
            lk_values[k - 1] /= k

        valid_k_values = lk_values > 0
        if torch.sum(valid_k_values) > 1:
            k_arr = torch.arange(1, self.max_k + 1, device=arr.device)[valid_k_values]
            log_k = torch.log(k_arr.float())
            log_lk = torch.log(lk_values[valid_k_values])
            slope, _ = torch.polyfit(log_k, log_lk, 1)
            return -slope.item()
        else:
            return 0.0

# ============================
# Residual Block
# ============================

class ResidualBlock(nn.Module):
    """Residual block with two linear layers and a skip connection."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = APELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.norm(out)
        out += residual
        out = self.activation(out)
        return out

# ============================
# Transformer and Attention Layers
# ============================

class TransformerEncoderLayerCustom(nn.Module):
    """Custom Transformer Encoder Layer with multi-headed attention and residual connections."""
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = F.relu
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: the sequence to the encoder (S, N, E)
            src_mask: the mask for the src sequence (S, S)
        """
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoderCustom(nn.Module):
    """Custom Transformer Encoder with multiple layers."""
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerCustom(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_mask)
        src = self.norm(src)
        return src

class MultiHeadAttentionCustom(nn.Module):
    """Multi-Head Attention Layer."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

# ============================
# MLP Classes with Enhancements
# ============================

class BaseMLP(nn.Module):
    """Base MLP with support for custom layers, normalization, and residual connections."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout_rate: float,
        use_custom_layers: bool,
        window_size: int,
        custom_layers: Optional[List[str]] = None,
        use_instance_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 8,
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_custom_layers = use_custom_layers
        self.use_instance_norm = use_instance_norm
        self.use_group_norm = use_group_norm
        self.use_residual = use_residual
        self.activation = APELU()
        self.dropout = nn.Dropout(dropout_rate)

        # Custom Layers
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

        # MLP Layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_blocks = nn.ModuleList() if use_residual else None

        prev_features = in_features
        for i in range(num_layers):
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(prev_features, out_features))
            if self.use_instance_norm:
                self.norms.append(nn.InstanceNorm1d(out_features))
            elif self.use_group_norm:
                self.norms.append(nn.GroupNorm(num_groups, out_features))
            else:
                self.norms.append(nn.LayerNorm(out_features))
            if self.use_residual and i < num_layers - 1:
                self.residual_blocks.append(ResidualBlock(out_features, hidden_dim, dropout_rate))
            prev_features = out_features

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with optional custom layers."""
        if self.use_custom_layers:
            outputs = []
            for cl in self.custom_layers_list:
                out = cl(x)
                outputs.append(out)
            x = torch.cat(outputs, dim=-1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.norms):
                if self.use_instance_norm:
                    x = self.norms[i](x.unsqueeze(2)).squeeze(2)
                elif self.use_group_norm:
                    x = self.norms[i](x.unsqueeze(2)).squeeze(2)
                else:
                    x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
                if self.use_residual and i < len(self.residual_blocks):
                    x = self.residual_blocks[i](x)
        return x

class AdaptiveModulationMLP(BaseMLP):
    """MLP with time-aware modulation, residual connections, and advanced normalization."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout_rate: float,
        time_encoding_dim: int,
        use_custom_layers: bool,
        window_size: int,
        custom_layers: Optional[List[str]] = None,
        use_instance_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 8,
        use_residual: bool = False,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            dropout_rate,
            use_custom_layers,
            window_size,
            custom_layers,
            use_instance_norm,
            use_group_norm,
            num_groups,
            use_residual
        )
        self.sinusoidal_encoding = SinusoidalTimeEncoding(time_encoding_dim)
        self.time_biases = nn.ModuleList([
            TimeAwareBias(hidden_dim, time_encoding_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.modulations = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_layers - 1)
        ])

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """Forward pass with time-aware modulation."""
        time_encoding = self.sinusoidal_encoding(time_step)

        if self.use_custom_layers:
            outputs = []
            for cl in self.custom_layers_list:
                out = cl(x)
                outputs.append(out)
            x = torch.cat(outputs, dim=-1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.norms):
                mod_factor = self.modulations[i] + self.time_biases[i](time_encoding)
                x = x * mod_factor
                if self.use_instance_norm:
                    x = self.norms[i](x.unsqueeze(2)).squeeze(2)
                elif self.use_group_norm:
                    x = self.norms[i](x.unsqueeze(2)).squeeze(2)
                else:
                    x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
                if self.use_residual and i < len(self.residual_blocks):
                    x = self.residual_blocks[i](x)
        return x

# ============================
# Sinusoidal Time Encoding
# ============================

class SinusoidalTimeEncoding(nn.Module):
    """Encodes time using sinusoidal functions."""
    def __init__(self, time_encoding_dim: int):
        super().__init__()
        self.time_encoding_dim = time_encoding_dim
        self.frequencies = 10**(torch.arange(0, time_encoding_dim//2) * (-2/(time_encoding_dim//2)))

    def forward(self, time_step: torch.Tensor) -> torch.Tensor:
        """Applies sinusoidal time encoding."""
        time_step = time_step.float().unsqueeze(-1)  # (batch_size, 1)
        scaled_time = time_step * self.frequencies.to(time_step.device)  # (batch_size, time_encoding_dim//2)
        sin_enc = torch.sin(scaled_time)
        cos_enc = torch.cos(scaled_time)
        if self.time_encoding_dim % 2 == 0:
            encoding = torch.cat([sin_enc, cos_enc], dim=-1)  # (batch_size, time_encoding_dim)
        else:
            zero_pad = torch.zeros_like(cos_enc[:, :1], device=cos_enc.device)
            encoding = torch.cat([sin_enc, cos_enc, zero_pad], dim=-1)
        return encoding

# ============================
# Time-Aware Bias
# ============================

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

# ============================
# Policy Distiller with Ensemble Methods
# ============================

class PolicyDistillerEnsemble(nn.Module):
    """Combines outputs from multiple specialist policies using an ensemble approach."""
    def __init__(self, specialist_policies: List[nn.Module], config: EnvironmentConfig):
        super().__init__()
        self.specialists = nn.ModuleList(specialist_policies)
        self.ensemble_size = config.ensemble_size
        self.mlp = nn.Linear(config.action_dim * self.ensemble_size * 2, config.action_dim * 2)  # For mu and log_sigma

    def forward(self, state: torch.Tensor, time_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch_size, seq_length, state_dim)
            time_step: (batch_size, )
        Returns:
            mu: (batch_size, action_dim)
            log_sigma: (batch_size, action_dim)
        """
        outputs = [spec(state, time_step) for spec in self.specialists]  # List of (batch_size, 2 * action_dim)
        mus = torch.stack([o[0] for o in outputs], dim=-1)  # (batch_size, action_dim, ensemble_size)
        log_sigmas = torch.stack([o[1] for o in outputs], dim=-1)  # (batch_size, action_dim, ensemble_size)

        # Concatenate along the ensemble dimension
        mus = mus.view(mus.size(0), -1)  # (batch_size, action_dim * ensemble_size)
        log_sigmas = log_sigmas.view(log_sigmas.size(0), -1)  # (batch_size, action_dim * ensemble_size)

        # Pass through an MLP to aggregate
        aggregated = self.mlp(torch.cat([mus, log_sigmas], dim=-1))  # (batch_size, action_dim * 2)
        mu = torch.tanh(aggregated[:, :config.action_dim])
        log_sigma = torch.clamp(aggregated[:, config.action_dim:], min=-20, max=2)
        return mu, log_sigma

    def compute_log_prob(self, mu: torch.Tensor, log_sigma: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of actions under the current policy."""
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return log_prob

# ============================
# High-Level Policy
# ============================

class HighLevelPolicy(nn.Module):
    """High-Level Policy Network for hierarchical decision-making."""
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs probability of selecting a high-level action."""
        return self.mlp(x)

# ============================
# Market Mode Classifier
# ============================

class MarketModeClassifier(nn.Module):
    """Classifies the current market mode."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs market mode logits."""
        return self.mlp(x)

# ============================
# Meta Controller
# ============================

class MetaController(nn.Module):
    """Meta Controller for dynamic hyperparameter adjustment."""
    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.state_dim + 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_hyperparams + 3),
            nn.Sigmoid()
        )
        self.num_hyperparams = config.num_hyperparams
        self.ema_smoothing = 0.9  # Smoothing factor

        self.register_buffer('ema_values', torch.zeros(config.num_hyperparams + 3))

    def forward(self, x: torch.Tensor, reward_stats: torch.Tensor) -> torch.Tensor:
        cat_input = torch.cat([x, reward_stats], dim=-1)
        out = self.mlp(cat_input)

        if self.ema_values is None:
            self.ema_values = out.detach()
        else:
            self.ema_values = self.ema_smoothing * self.ema_values + (1 - self.ema_smoothing) * out.detach()

        return self.ema_values

# File: models/components.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SphericalLayer(nn.Module):
    """Base layer for spherical operations"""
    def __init__(self, input_size: int, output_size: int, epsilon: float = 1e-8):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon

        # Initialize weights with modified Xavier/Glorot
        scale = (6.0 / (input_size + output_size)) ** 0.5
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * scale)
        self.scale_factor = nn.Parameter(torch.ones(output_size))

        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to unit hypersphere"""
        with torch.no_grad():
            self.weight.div_(torch.norm(self.weight, dim=1, keepdim=True) + self.epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with scaled normalized weights"""
        self._normalize_weights()
        return self.scale_factor.unsqueeze(0) * F.linear(x, self.weight)

class SLERPLayer(nn.Module):
    """Spherical Linear Interpolation Layer"""
    def __init__(self, size: int, epsilon: float = 1e-8, use_slerp: bool = True):
        super().__init__()
        self.size = size
        self.epsilon = epsilon
        self.use_slerp = use_slerp

        # Trainable eigen learning rate
        self.alpha_raw = nn.Parameter(torch.zeros(size))

    def forward(self, h_t: torch.Tensor, h_new: torch.Tensor) -> torch.Tensor:
        """Apply SLERP or NLERP"""
        alpha = torch.sigmoid(self.alpha_raw)

        if not self.use_slerp:
            # NLERP (Normalized Linear Interpolation)
            return F.normalize(
                (1 - alpha) * h_t + alpha * h_new,
                dim=-1,
                eps=self.epsilon
            )

        # SLERP implementation
        h_t = F.normalize(h_t, dim=-1, eps=self.epsilon)
        h_new = F.normalize(h_new, dim=-1, eps=self.epsilon)

        dot_product = torch.sum(h_t * h_new, dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1 + self.epsilon, 1 - self.epsilon)

        theta = torch.arccos(dot_product)
        sin_theta = torch.sin(theta)

        # Handle the case when sin(theta) is very small
        sin_theta = sin_theta + self.epsilon

        h_t_coeff = torch.sin((1 - alpha) * theta) / sin_theta
        h_new_coeff = torch.sin(alpha * theta) / sin_theta

        return F.normalize(
            h_t_coeff * h_t + h_new_coeff * h_new,
            dim=-1,
            eps=self.epsilon
        )

class RecurrentSphericalLayer(nn.Module):
    """Recurrent layer with spherical normalization"""
    def __init__(self, hidden_size: int, epsilon: float = 1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon

        # Initialize close to identity
        eye = torch.eye(hidden_size)
        noise = torch.randn(hidden_size, hidden_size) * 0.01
        self.weight = nn.Parameter(eye + noise)
        self.scale_factor = nn.Parameter(torch.ones(hidden_size))

        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to unit hypersphere"""
        with torch.no_grad():
            self.weight.div_(torch.norm(self.weight, dim=1, keepdim=True) + self.epsilon)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass with scaled normalized weights"""
        self._normalize_weights()
        return self.scale_factor.unsqueeze(0) * F.linear(h, self.weight)

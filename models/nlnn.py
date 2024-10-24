import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Tuple, Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class TradingConfig:
    input_size: int = 5  # OHLCV features
    hidden_size: int = 256
    output_size: int = 3  # Buy, Sell, Hold
    num_layers: int = 2
    sequence_length: int = 128
    batch_size: int = 512
    learning_rate: float = 1e-3
    epsilon: float = 1e-8
    dropout: float = 0.1
    use_slerp: bool = True
    use_mixed_precision: bool = True

class SphericalLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, epsilon: float = 1e-8):
        super().__init__()
        scale = (6.0 / (input_size + output_size)) ** 0.5
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * scale)
        self.scale = nn.Parameter(torch.ones(output_size))
        self.epsilon = epsilon
        self._normalize_weights()

    def _normalize_weights(self):
        with torch.no_grad():
            norm = torch.norm(self.weight, dim=1, keepdim=True)
            self.weight.div_(norm + self.epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._normalize_weights()
        return self.scale.unsqueeze(0) * F.linear(x, self.weight)

class SLERP(nn.Module):
    def __init__(self, size: int, epsilon: float = 1e-8):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(size))
        self.epsilon = epsilon

    def forward(self, h_t: torch.Tensor, h_new: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha)
        h_t = F.normalize(h_t, dim=-1, eps=self.epsilon)
        h_new = F.normalize(h_new, dim=-1, eps=self.epsilon)
        
        dot_product = torch.sum(h_t * h_new, dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1 + self.epsilon, 1 - self.epsilon)
        
        theta = torch.arccos(dot_product)
        sin_theta = torch.sin(theta) + self.epsilon
        
        h_t_coeff = torch.sin((1 - alpha) * theta) / sin_theta
        h_new_coeff = torch.sin(alpha * theta) / sin_theta
        
        return F.normalize(h_t_coeff * h_t + h_new_coeff * h_new, dim=-1, eps=self.epsilon)

class TradingNLNN(nn.Module):
    def __init__(self, config: TradingConfig):
        super().__init__()
        self.config = config
        
        # Hardware optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler(enabled=config.use_mixed_precision)
        
        # Network layers
        self.input_norm = nn.LayerNorm(config.input_size)
        self.input_layer = SphericalLinear(config.input_size, config.hidden_size)
        
        self.recurrent_layers = nn.ModuleList([
            SphericalLinear(config.hidden_size, config.hidden_size)
            for _ in range(config.num_layers)
        ])
        
        self.slerp_layers = nn.ModuleList([
            SLERP(config.hidden_size)
            for _ in range(config.num_layers)
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        self.output_layer = SphericalLinear(config.hidden_size, config.output_size)
        
        # Training components
        self.eta = nn.Parameter(torch.ones(1))
        
        self.to(self.device)
        self._optimize_memory_format()

    def _optimize_memory_format(self):
        self.to(memory_format=torch.channels_last)
        for param in self.parameters():
            if param.dim() == 4:
                param.data = param.data.to(memory_format=torch.channels_last)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=self.config.epsilon)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return self._normalize(
            torch.randn(self.config.num_layers, batch_size, self.config.hidden_size, 
                       device=self.device)
        )

    @autocast(device_type='cuda')
    def forward(self, x: torch.Tensor, h_init: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape[:2]
        h_t = h_init if h_init is not None else self.init_hidden(batch_size)
        
        outputs = []
        for t in range(seq_len):
            x_t = self.input_norm(x[:, t])
            input_proj = self.input_layer(x_t)
            
            for layer_idx in range(self.config.num_layers):
                recurrent_proj = self.recurrent_layers[layer_idx](h_t[layer_idx])
                h_new = self._normalize(input_proj + recurrent_proj)
                
                if self.config.use_slerp:
                    h_t[layer_idx] = self.slerp_layers[layer_idx](h_t[layer_idx], h_new)
                else:
                    h_t[layer_idx] = h_new
                
                h_t[layer_idx] = self.dropout(h_t[layer_idx])
            
            output = self.output_layer(h_t[-1])
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, h_t

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate * torch.sigmoid(self.eta),
            eps=self.config.epsilon,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

class TradingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_loss(pred.view(-1, pred.size(-1)), target.view(-1))

def create_dataloaders(config: TradingConfig, train_data: torch.Tensor, val_data: torch.Tensor):
    train_dataset = torch.utils.data.TensorDataset(
        train_data[:, :-1], train_data[:, 1:]  # Input, target pairs
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_data[:, :-1], val_data[:, 1:]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=12,
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        pin_memory=True,
        num_workers=12,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def optimize_hardware():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        
    torch.set_num_threads(24)  # Optimize for Ryzen 9 7900X
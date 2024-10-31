import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class LN2Config:
    """Enhanced configuration for LN² model with hardware optimization settings"""
    feature_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    sequence_length: int
    batch_size: int
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epsilon: float = 1e-8
    eigen_adaptation_rate: float = 0.01
    slerp_rate: float = 0.1
    use_mixed_precision: bool = True
    use_slerp: bool = True

class SphericalLinear(nn.Module):
    """Optimized linear layer with weight normalization on hypersphere"""
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

class OptimizedSLERP(nn.Module):
    """Enhanced SLERP implementation with numerical stability"""
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

class OptimizedLiquidCell(nn.Module):
    """Hardware-optimized liquid neural network cell"""
    def __init__(self, input_dim: int, hidden_dim: int, config: LN2Config):
        super().__init__()
        self.config = config
        
        self.input_proj = SphericalLinear(input_dim, hidden_dim)
        self.state_proj = SphericalLinear(hidden_dim, hidden_dim)
        self.slerp = OptimizedSLERP(hidden_dim, config.epsilon)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        if h is None:
            h = F.normalize(
                torch.zeros(batch_size, self.state_proj.weight.size(0), 
                          device=x.device),
                dim=-1,
                eps=self.config.epsilon
            )
        
        input_proj = self.input_proj(x)
        state_proj = self.state_proj(h)
        
        h_new = F.normalize(input_proj + state_proj, dim=-1, eps=self.config.epsilon)
        
        if self.config.use_slerp:
            h_next = self.slerp(h, h_new)
        else:
            h_next = h_new
            
        return self.norm(h_next), h_next

class OptimizedLN2Model(nn.Module):
    """Hardware-optimized Liquid Normalized Neural Network model"""
    def __init__(self, config: LN2Config):
        super().__init__()
        self.config = config
        
        # Hardware optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler(enabled=config.use_mixed_precision)
        
        # Network layers
        self.input_norm = nn.LayerNorm(config.feature_dim)
        self.embedding = SphericalLinear(config.feature_dim, config.hidden_dim)
        
        self.liquid_cells = nn.ModuleList([
            OptimizedLiquidCell(config.hidden_dim, config.hidden_dim, config)
            for _ in range(config.num_layers)
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        self.output_layer = SphericalLinear(config.hidden_dim, config.output_dim)
        
        # Adaptive learning rate
        self.eta = nn.Parameter(torch.ones(1))
        
        self.to(self.device)
        self._optimize_memory_format()

    def _optimize_memory_format(self):
        """Optimize memory layout for better performance"""
        self.to(memory_format=torch.channels_last)
        for param in self.parameters():
            if param.dim() == 4:
                param.data = param.data.to(memory_format=torch.channels_last)

    def init_states(self, batch_size: int) -> List[torch.Tensor]:
        return [F.normalize(
            torch.randn(batch_size, self.config.hidden_dim, device=self.device),
            dim=-1,
            eps=self.config.epsilon
        ) for _ in range(self.config.num_layers)]

    @autocast(device_type='cuda')
    def forward(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len = x.shape[:2]
        
        if states is None:
            states = self.init_states(batch_size)
        
        outputs = []
        for t in range(seq_len):
            x_t = self.input_norm(x[:, t])
            h = self.embedding(x_t)
            
            new_states = []
            for i, cell in enumerate(self.liquid_cells):
                h, new_state = cell(h, states[i])
                h = self.dropout(h)
                new_states.append(new_state)
            
            states = new_states
            output = self.output_layer(h)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, states

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer with adaptive learning rate"""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate * torch.sigmoid(self.eta),
            eps=self.config.epsilon,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

class LN2Loss(nn.Module):
    """Loss function for LN² model with regularization"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, states: List[torch.Tensor]) -> torch.Tensor:
        pred_loss = self.criterion(pred, target)
        
        # State transition regularization
        state_reg = sum(
            torch.norm(s[1:] - s[:-1], dim=-1).mean() 
            for s in states
        ) / len(states)
        
        return pred_loss + 0.1 * state_reg

def optimize_hardware():
    """Optimize hardware settings for maximum performance"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
    
    torch.set_num_threads(24)  # Optimize for modern CPUs

def create_dataloaders(config: LN2Config, train_data: torch.Tensor, val_data: torch.Tensor):
    """Create optimized dataloaders for training"""
    train_dataset = torch.utils.data.TensorDataset(
        train_data[:, :-1], train_data[:, 1:]
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
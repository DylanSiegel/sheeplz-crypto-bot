import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

@dataclass
class NLNNConfig:
    hidden_size: int = 256
    input_size: int = 128
    output_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    epsilon: float = 1e-8
    use_mixed_precision: bool = True
    num_threads: int = 24  
    batch_size: int = 512
    sequence_length: int = 128
    grad_clip: float = 1.0
    learning_rate: float = 1e-3

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text: str, seq_length: int):
        chars = sorted(list(set(text)))
        self.char_to_idx: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_length = seq_length
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data) - self.seq_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y

    def decode(self, indices: torch.Tensor) -> str:
        return ''.join(self.idx_to_char[idx.item()] for idx in indices)

class SphericalOptimizer(torch.optim.Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = 0.9, 0.999

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
                p.div_(torch.norm(p) + group['eps'])

        return loss

class NLNN(nn.Module):
    def __init__(self, config: NLNNConfig):
        super().__init__()
        self.config = config
        
        torch.set_num_threads(config.num_threads)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler(enabled=config.use_mixed_precision)
        
        # Enhanced weight initialization with per-neuron scaling
        self.input_weights = nn.Parameter(torch.randn(config.input_size, config.hidden_size))
        self.recurrent_weights = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))
        self.output_weights = nn.Parameter(torch.randn(config.hidden_size, config.output_size))
        
        # Per-neuron scaling factors
        self.lambda_i = nn.Parameter(torch.randn(config.hidden_size))
        self.lambda_r = nn.Parameter(torch.randn(config.hidden_size))
        self.s_z = nn.Parameter(torch.randn(1))
        
        # Eigen learning rates (alpha) with raw parameters
        self.alpha_raw = nn.Parameter(torch.randn(config.hidden_size))
        
        # Global learning rate scaling
        self.eta_scale = nn.Parameter(torch.randn(1))
        
        self._normalize_parameters()
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer('epsilon', torch.tensor(config.epsilon))
        
    def _normalize_parameters(self):
        with torch.no_grad():
            self.input_weights.div_(torch.norm(self.input_weights, dim=0, keepdim=True) + self.epsilon)
            self.recurrent_weights.div_(torch.norm(self.recurrent_weights, dim=0, keepdim=True) + self.epsilon)
            self.output_weights.div_(torch.norm(self.output_weights, dim=0, keepdim=True) + self.epsilon)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / (torch.norm(x, dim=-1, keepdim=True) + self.epsilon)

    def slerp(self, h_t: torch.Tensor, h_new: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        h_t = self.normalize(h_t)
        h_new = self.normalize(h_new)
        
        dot_product = torch.sum(h_t * h_new, dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1 + self.epsilon, 1 - self.epsilon)
        
        theta = torch.arccos(dot_product)
        sin_theta = torch.sin(theta)
        
        mask = (sin_theta > self.epsilon).float()
        
        h_t_coeff = torch.sin((1 - alpha) * theta) / (sin_theta + self.epsilon)
        h_new_coeff = torch.sin(alpha * theta) / (sin_theta + self.epsilon)
        
        result = mask * (h_t_coeff * h_t + h_new_coeff * h_new) + (1 - mask) * h_new
        return self.normalize(result)

    @autocast()
    def forward(self, x: torch.Tensor, h_init: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 2:  # Handle both one-hot and index inputs
            x = F.one_hot(x, num_classes=self.config.input_size).float()
        
        batch_size, seq_len = x.shape[:2]
        
        if h_init is None:
            h_t = self.normalize(torch.randn(batch_size, self.config.hidden_size, device=self.device))
        else:
            h_t = h_init
            
        outputs = []
        alpha = torch.sigmoid(self.alpha_raw)
        
        for t in range(seq_len):
            x_t = x[:, t]
            
            # Apply per-neuron scaling
            input_proj = self.lambda_i.unsqueeze(0) * F.linear(x_t, self.input_weights.T)
            recurrent_proj = self.lambda_r.unsqueeze(0) * F.linear(h_t, self.recurrent_weights.T)
            
            h_new = self.normalize(input_proj + recurrent_proj)
            h_t = self.slerp(h_t, h_new, alpha)
            h_t = self.dropout(h_t)
            
            output = self.s_z * F.linear(h_t, self.output_weights.T)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, h_t

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return self.normalize(torch.randn(batch_size, self.config.hidden_size, device=self.device))

class NLNNTrainer:
    def __init__(self, model: NLNN, config: NLNNConfig):
        self.model = model
        self.config = config
        self.optimizer = SphericalOptimizer(model.parameters(), {
            'lr': config.learning_rate, 
            'eps': config.epsilon
        })
        self.criterion = nn.CrossEntropyLoss()
        
    @torch.cuda.amp.autocast()
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        outputs, _ = self.model(x)
        outputs = outputs.view(-1, outputs.size(-1))
        y = y.view(-1)
        loss = self.criterion(outputs, y)
        return loss
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = [b.to(self.model.device) for b in batch]
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                loss = self.training_step(batch)
            
            self.model.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.model.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.model.scaler.step(self.optimizer)
            self.model.scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
            
        return total_loss / len(dataloader)

    def generate_text(self, dataset: TextDataset, seed_text: str, length: int = 100, temperature: float = 1.0) -> str:
        self.model.eval()
        
        # Convert seed text to indices
        indices = torch.tensor([dataset.char_to_idx[c] for c in seed_text], dtype=torch.long)
        indices = indices.unsqueeze(0).to(self.model.device)  # Add batch dimension
        
        # Initialize hidden state
        h_t = self.model.init_hidden(1)
        
        generated_indices = []
        
        with torch.no_grad():
            # Process seed text
            outputs, h_t = self.model(indices, h_t)
            
            # Generate new characters
            for _ in range(length):
                output = outputs[:, -1, :] / temperature
                probs = F.softmax(output, dim=-1)
                next_char_idx = torch.multinomial(probs, 1)
                
                generated_indices.append(next_char_idx.item())
                
                # Prepare input for next iteration
                next_input = F.one_hot(next_char_idx, num_classes=self.config.output_size).float()
                outputs, h_t = self.model(next_input, h_t)
        
        # Convert generated indices back to text
        generated_text = dataset.decode(torch.tensor(generated_indices))
        return seed_text + generated_text

def create_model(config: NLNNConfig) -> NLNN:
    model = NLNN(config)
    model = model.to(model.device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model
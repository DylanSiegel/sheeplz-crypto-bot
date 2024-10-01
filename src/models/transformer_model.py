import numpy as np
import torch
import torch.nn as nn
import math
from typing import Tuple
from src.models.base_model import BaseModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class TransformerModel(BaseModel):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, output_size: int):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.input_linear = nn.Linear(input_size, d_model)
        self.output_linear = nn.Linear(d_model, output_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: (batch_size, sequence_length, input_size)
        src = self.input_linear(src).permute(1, 0, 2)  # (sequence_length, batch_size, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # Global average pooling
        output = self.output_linear(output)  # (batch_size, output_size)
        return output

    def get_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, sequence_length, input_size)
            q_values = self(state_tensor)
            return q_values.argmax().item()

    def update(self, optimizer: torch.optim.Optimizer, criterion: nn.Module, 
               state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        optimizer.zero_grad()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, sequence_length, input_size)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        q_values = self(state_tensor)
        next_q_values = self(next_state_tensor)
        
        current_q_value = q_values[0][action]
        next_q_value = next_q_values.max()
        
        expected_q_value = reward + (0.99 * next_q_value * (1 - int(done)))
        
        loss = criterion(current_q_value, expected_q_value.detach())
        loss.backward()
        optimizer.step()
        
        return loss.item()
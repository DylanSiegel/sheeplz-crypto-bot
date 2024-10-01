import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.models.base_model import BaseModel

class TimesBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(TimesBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.ln(x + residual)
        return x

class TimesNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(TimesNet, self).__init__()
        self.blocks = nn.ModuleList([
            TimesBlock(input_size if i == 0 else hidden_size, hidden_size) 
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

class TimesNetModel(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(TimesNetModel, self).__init__()
        self.timesnet = TimesNet(input_size, hidden_size, num_layers, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x shape: (batch_size, sequence_length, input_size)
        return self.timesnet(x)  # output shape: (batch_size, output_size)

    def get_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, input_size)
            q_values = self(state_tensor)
            return q_values.argmax().item()

    def update(self, optimizer: torch.optim.Optimizer, criterion: nn.Module, 
               state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        optimizer.zero_grad()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, input_size)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
        
        q_values = self(state_tensor)
        next_q_values = self(next_state_tensor)
        
        current_q_value = q_values[0][action]
        next_q_value = next_q_values.max()
        
        expected_q_value = reward + (0.99 * next_q_value * (1 - int(done)))
        
        loss = criterion(current_q_value, expected_q_value.detach())
        loss.backward()
        optimizer.step()
        
        return loss.item()
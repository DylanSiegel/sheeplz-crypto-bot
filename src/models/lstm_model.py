# File: src/models/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.base_model import BaseModel

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, hidden_states):
        attention_weights = F.softmax(self.attention(hidden_states), dim=1)
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        return context_vector, attention_weights

class LSTMModel(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = AttentionLayer(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        context_vector, attention_weights = self.attention(lstm_out)
        out = self.batch_norm(context_vector)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out, attention_weights

    def get_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, input_size)
            q_values, _ = self(state_tensor)
            return q_values.argmax().item()

    def update(self, optimizer: torch.optim.Optimizer, criterion: nn.Module, 
               state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        optimizer.zero_grad()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, input_size)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
        
        q_values, _ = self(state_tensor)
        next_q_values, _ = self(next_state_tensor)
        
        current_q_value = q_values[0][action]
        next_q_value = next_q_values.max()
        
        expected_q_value = reward + (0.99 * next_q_value * (1 - int(done)))
        
        loss = criterion(current_q_value, expected_q_value.detach())
        loss.backward()
        optimizer.step()
        
        return loss.item()

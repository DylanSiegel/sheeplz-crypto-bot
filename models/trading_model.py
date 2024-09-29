# File: models/trading_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TradingModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(TradingModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

    def get_action(self, state: np.ndarray) -> float:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            q_values = self(state_tensor)
            return q_values.squeeze().item()

    def update(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
        action_tensor = torch.FloatTensor([action])
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([float(done)])

        q_values = self(state_tensor)
        next_q_values = self(next_state_tensor)

        target = reward_tensor + (1 - done_tensor) * 0.99 * torch.max(next_q_values)
        loss = self.criterion(q_values.squeeze(), target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
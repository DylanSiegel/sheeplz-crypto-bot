# File: models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TradingModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(TradingModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Assuming x shape: (batch, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Taking the output of the last time step
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out

    def get_action(self, state):
        with torch.no_grad():
            q_values = self(torch.FloatTensor(state).unsqueeze(0))
            return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        # Implement your update logic here (e.g., Q-learning updates)
        pass

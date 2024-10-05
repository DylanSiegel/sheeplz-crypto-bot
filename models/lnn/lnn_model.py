# File: models/lnn/lnn_model.py

import torch
import torch.nn as nn

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2, dropout: float = 0.2):
        super(LiquidNeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LNN.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_size)
        out = self.dropout(out[:, -1, :])  # Take the output from the last time step
        out = self.relu(out)
        out = self.fc(out)
        return out

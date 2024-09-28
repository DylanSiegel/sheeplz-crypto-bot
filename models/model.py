# File: models/model.py

import torch
import torch.nn as nn

class TradingModel(nn.Module):
    """
    Deep reinforcement learning model for trading.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(TradingModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass input through LSTM
        out, _ = self.lstm(x)

        # Take the output from the last timestep
        out = out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)

        return out

# Example usage
# model = TradingModel(input_size=10, hidden_size=64, output_size=3)  # 3 output neurons for 3 actions
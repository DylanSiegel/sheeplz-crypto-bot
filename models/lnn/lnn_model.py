# models/lnn/lnn_model.py

import torch
import torch.nn as nn

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.liquid_layer = nn.RNN(input_size, hidden_size, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.liquid_layer(x)
        out = self.fc(out[:, -1, :])
        return out

# File: models/lnn_model.py
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, hidden_size):
        super(ODEFunc, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.nonlinearity = nn.LeakyReLU()

    def forward(self, t, x):
        return self.nonlinearity(self.linear(x))

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.ode_func = ODEFunc(hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        t = torch.tensor([0.0, 1.0], dtype=torch.float32).to(x.device)
        x = self.input_layer(x)
        # Adjusted solver options to prevent underflow error
        ode_sol = odeint(
            self.ode_func,
            x,
            t,
            method='rk4',  # Using a fixed-step solver
            options={'step_size': 0.1}  # Specify a suitable step size
        )
        x = ode_sol[-1]
        x = self.output_layer(x)
        return x

# File: models/lnn_model.py
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint  # Using adjoint method for memory efficiency

class ODEFunc(nn.Module):
    def __init__(self, hidden_size):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, t, x):
        return self.net(x)

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_ode_layers=5):
        super(LiquidNeuralNetwork, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.ode_funcs = nn.ModuleList([ODEFunc(hidden_size) for _ in range(num_ode_layers)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh(),  # Output between -1 and 1
        )
        self.res_connection = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # Time tensor with multiple points for higher resolution
        t = torch.linspace(0, 1, steps=10).to(x.device)
        residual = self.res_connection(x)
        x = self.input_layer(x)
        x = x + residual  # Adding residual connection

        # Pass through multiple ODE layers
        for ode_func in self.ode_funcs:
            x = odeint(ode_func, x, t, method='dopri5')[-1]

        x = self.output_layer(x)
        return x

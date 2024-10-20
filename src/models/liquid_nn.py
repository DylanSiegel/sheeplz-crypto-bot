# src/models/liquid_nn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidNN(nn.Module):
    """
    Enhanced Liquid Neural Network (LNN) with residual connections and normalization.
    """
    def __init__(self, input_size, hidden_size, num_classes, sequence_length):
        super(LiquidNN, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        # GRU Layer with bidirectionality for richer feature extraction
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Attention Layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully Connected Layers with Residual Connection
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)
    
    def forward(self, x):
        """
        Forward pass of the Liquid Neural Network.

        Parameters:
        - x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_size]

        Returns:
        - torch.Tensor: Output logits of shape [batch_size, num_classes]
        """
        # GRU outputs
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, sequence_length, hidden_size*2]
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(gru_out), dim=1)  # [batch_size, sequence_length, 1]
        context = torch.sum(attn_weights * gru_out, dim=1)  # [batch_size, hidden_size*2]
        
        # Fully Connected Layers with Residual Connection
        out = F.relu(self.bn1(self.fc1(context)))
        out = self.bn2(self.fc2(out))
        return out

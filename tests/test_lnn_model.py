# File: tests/test_lnn_model.py

import pytest
import torch
from models.lnn.lnn_model import LiquidNeuralNetwork

def test_lnn_forward():
    input_size = 5
    hidden_size = 10
    output_size = 1
    batch_size = 2
    seq_len = 1

    model = LiquidNeuralNetwork(input_size, hidden_size, output_size)
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    output = model(dummy_input)

    # Check output shape
    assert output.shape == (batch_size, output_size)

def test_lnn_training_step():
    input_size = 5
    hidden_size = 10
    output_size = 1

    model = LiquidNeuralNetwork(input_size, hidden_size, output_size)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Dummy data
    X = torch.randn(4, 1, input_size)
    y = torch.tensor([1, 0, 1, 0], dtype=torch.float32)

    # Forward pass
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Ensure loss is a scalar
    assert loss.item() >= 0

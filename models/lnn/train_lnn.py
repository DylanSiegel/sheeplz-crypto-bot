# models/lnn/train_lnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from models.lnn.lnn_model import LiquidNeuralNetwork
from torch.utils.data import DataLoader, TensorDataset

def train_lnn():
    # Load distilled dataset
    data = pd.read_csv("data/distilled/distilled_data.csv")
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, optimizer
    model = LiquidNeuralNetwork(input_size=X.shape[1], hidden_size=64, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch.unsqueeze(1))
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "models/lnn/lnn_model.pth")

if __name__ == "__main__":
    train_lnn()

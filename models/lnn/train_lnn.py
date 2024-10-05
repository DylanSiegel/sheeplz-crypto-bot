# File: models/lnn/train_lnn.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from models.lnn.lnn_model import LiquidNeuralNetwork
from models.utils.config import Config
from sklearn.preprocessing import MinMaxScaler
import logging


def train_lnn():
    """Trains the LNN model and saves it."""
    config = Config("configs/config.yaml")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Load and prepare your data
        # Replace 'your_training_data.csv' with your actual data source
        data = pd.read_csv("data/your_training_data.csv")  # Ensure this file exists and is correctly formatted
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Scale the input features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)  # Shape: (batch_size, seq_len, input_size)
        y_tensor = torch.tensor(y, dtype=torch.float32)  # Shape: (batch_size,)

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize the model, loss function, and optimizer
        input_size = X_scaled.shape[1]
        model = LiquidNeuralNetwork(input_size, config.lnn_hidden_size, 1)
        criterion = nn.BCEWithLogitsLoss()  # Suitable for binary classification
        optimizer = optim.Adam(model.parameters(), lr=config.lnn_learning_rate)

        # Training loop
        epochs = config.lnn_training_epochs
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X.half()).squeeze()  # Forward pass
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save the trained model
        torch.save(model.state_dict(), config.lnn_model_path)
        logging.info(f"LNN model trained and saved to {config.lnn_model_path}")

    except FileNotFoundError:
        logging.error("Training data file not found. Please provide a valid CSV file.")
    except Exception as e:
        logging.error(f"Error during LNN training: {e}")


if __name__ == "__main__":
    train_lnn()

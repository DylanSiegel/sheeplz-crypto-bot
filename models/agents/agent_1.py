# models/agents/agent.py

import torch
import numpy as np
from models.lnn.lnn_model import LiquidNeuralNetwork

class TradingAgent:
    def __init__(self, time_frame):
        self.time_frame = time_frame
        self.model = LiquidNeuralNetwork(input_size=INPUT_SIZE, hidden_size=64, output_size=1)
        self.model.load_state_dict(torch.load("models/lnn/lnn_model.pth"))
        self.model.eval()
        self.position = None  # 'long', 'short', or None

    def make_decision(self, market_data):
        # Prepare data for the model
        X = self.prepare_input(market_data)
        with torch.no_grad():
            output = self.model(X.unsqueeze(0).unsqueeze(0))
            prediction = output.item()

        # Simple decision logic based on prediction
        if prediction > THRESHOLD_BUY:
            if self.position != 'long':
                self.enter_position('long')
        elif prediction < THRESHOLD_SELL:
            if self.position != 'short':
                self.enter_position('short')
        else:
            self.exit_position()

    def prepare_input(self, market_data):
        # Extract features from market_data for the model
        features = np.array([...])  # Replace with actual feature extraction
        return torch.tensor(features, dtype=torch.float32)

    def enter_position(self, position_type):
        # Code to execute trade via MEXC API
        print(f"Entering {position_type} position.")
        self.position = position_type

    def exit_position(self):
        if self.position is not None:
            # Code to exit trade via MEXC API
            print(f"Exiting {self.position} position.")
            self.position = None

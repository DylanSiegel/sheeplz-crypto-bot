# File: data/data_processor.py
import asyncio
from typing import Dict, Any, List
from error_handler import ErrorHandler
import logging
from models.lnn_model import LiquidNeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Constants for LNN
INPUT_SIZE = 100  # Adjust based on your selected features
HIDDEN_SIZE = 256  # Should match the hidden_size in the model
OUTPUT_SIZE = 1

class DataProcessor:
    def __init__(
        self,
        data_queue: asyncio.Queue,
        error_handler: ErrorHandler,
        symbols: List[str],
        timeframes: List[str]
    ):
        self.data_queue = data_queue
        self.error_handler = error_handler
        self.symbols = symbols
        self.timeframes = timeframes
        self.logger = logging.getLogger("DataProcessor")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lnn = LiquidNeuralNetwork(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            num_ode_layers=5
        ).to(self.device)
        # Precomputed scaling parameters (replace with actual means and stds)
        self.feature_means = torch.zeros(INPUT_SIZE).to(self.device)
        self.feature_stds = torch.ones(INPUT_SIZE).to(self.device)

        # Batch processing parameters
        self.batch_size = 64  # Adjust based on your GPU capacity
        self.input_buffer = []
        self.target_buffer = []

        # Initialize optimizer and loss function for online training
        self.optimizer = optim.Adam(self.lnn.parameters(), lr=1e-4)
        self.loss_function = nn.MSELoss()

    def preprocess_data(self, data: Dict[str, Any]) -> torch.Tensor:
        """Preprocess data and perform feature engineering."""
        raw_features = self.extract_raw_features(data)
        if raw_features is None:
            return None

        # Perform complex feature engineering
        try:
            features = self._engineer_features(raw_features)
        except Exception as e:
            self.error_handler.handle_error(
                f"Feature Engineering Error: {e}",
                exc_info=True,
                data=data
            )
            return None

        # Ensure features match INPUT_SIZE
        if len(features) < INPUT_SIZE:
            padding = torch.zeros(INPUT_SIZE - len(features)).to(self.device)
            features = torch.cat([features, padding])
        elif len(features) > INPUT_SIZE:
            features = features[:INPUT_SIZE]

        # Scale features
        features_scaled = (features - self.feature_means) / (self.feature_stds + 1e-8)
        return features_scaled

    def extract_raw_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract raw features from data."""
        try:
            kline_data = data.get('d', {}).get('k')
            if kline_data:
                features = [
                    float(kline_data['o']),  # Open price
                    float(kline_data['c']),  # Close price
                    float(kline_data['h']),  # High price
                    float(kline_data['l']),  # Low price
                    float(kline_data['v']),  # Volume
                    # Add more raw features as needed
                ]
                return torch.tensor(features, dtype=torch.float32).to(self.device)
            else:
                return None
        except Exception as e:
            self.error_handler.handle_error(
                f"Feature Extraction Error: {e}",
                exc_info=True,
                data=data
            )
            return None

    def _engineer_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Perform complex feature engineering."""
        # Ensure raw_features is a 1D tensor on the correct device
        raw_features = raw_features.to(self.device)
        if raw_features.dim() == 0:
            raw_features = raw_features.unsqueeze(0)
        elif raw_features.dim() > 1:
            raw_features = raw_features.view(-1)

        features = [raw_features]

        # Rolling statistics
        if raw_features.numel() >= 5:
            rolling_mean = torch.mean(raw_features[-5:])
            rolling_std = torch.std(raw_features[-5:])
        else:
            rolling_mean = torch.mean(raw_features)
            rolling_std = torch.std(raw_features)

        rolling_mean = rolling_mean.unsqueeze(0)
        rolling_std = rolling_std.unsqueeze(0)
        features.extend([rolling_mean, rolling_std])

        # Technical indicators
        rsi = self._calculate_rsi(raw_features)
        macd = self._calculate_macd(raw_features)
        features.extend([rsi, macd])

        # Combine all features
        features = torch.cat(features)
        return features

    def _calculate_rsi(self, data: torch.Tensor, period: int = 14) -> torch.Tensor:
        """Calculate the Relative Strength Index (RSI)."""
        if data.numel() < period + 1:
            return torch.tensor([50.0], device=self.device)  # Default value when insufficient data

        # Compute price differences
        delta = data[1:] - data[:-1]
        delta = delta.to(self.device)

        # Separate gains and losses
        gains = torch.where(delta > 0, delta, torch.tensor(0.0, device=self.device))
        losses = torch.where(delta < 0, -delta, torch.tensor(0.0, device=self.device))

        # Calculate average gains and losses
        avg_gain = torch.mean(gains[-period:])
        avg_loss = torch.mean(losses[-period:])

        if avg_loss == 0:
            rs = torch.tensor(float('inf'), device=self.device)
        else:
            rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))
        return rsi.unsqueeze(0)

    def _calculate_macd(self, data: torch.Tensor, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> torch.Tensor:
        """Calculate the Moving Average Convergence Divergence (MACD)."""
        if data.numel() < slow_period:
            return torch.tensor([0.0], device=self.device)  # Default value when insufficient data

        # Calculate exponential moving averages (EMAs)
        ema_fast = self._calculate_ema(data, fast_period)
        ema_slow = self._calculate_ema(data, slow_period)
        macd_line = ema_fast - ema_slow

        if macd_line.numel() < signal_period:
            return torch.tensor([0.0], device=self.device)

        signal_line = self._calculate_ema(macd_line, signal_period)
        macd_histogram = macd_line[-1] - signal_line[-1]

        return macd_histogram.unsqueeze(0)

    def _calculate_ema(self, data: torch.Tensor, period: int) -> torch.Tensor:
        """Calculate Exponential Moving Average (EMA)."""
        alpha = 2 / (period + 1)
        ema = [data[0]]
        for price in data[1:]:
            ema_value = alpha * price + (1 - alpha) * ema[-1]
            ema.append(ema_value)
        return torch.stack(ema).to(self.device)

    def generate_target(self, data: Dict[str, Any]) -> torch.Tensor:
        """Generate target value for training."""
        # For demonstration, we'll use the future close price as the target
        try:
            future_kline_data = data.get('d', {}).get('k')  # Replace with actual future data
            if future_kline_data:
                future_close_price = float(future_kline_data['c'])
                return torch.tensor([future_close_price], dtype=torch.float32).to(self.device)
            else:
                return None
        except Exception as e:
            self.error_handler.handle_error(
                f"Target Generation Error: {e}",
                exc_info=True,
                data=data
            )
            return None

    async def run_lnn(self):
        """Continuously process data through the LNN and perform online training."""
        while True:
            try:
                data = await self.data_queue.get()
                input_tensor = self.preprocess_data(data)
                target_tensor = self.generate_target(data)
                if input_tensor is not None and target_tensor is not None:
                    self.input_buffer.append(input_tensor)
                    self.target_buffer.append(target_tensor)

                if len(self.input_buffer) >= self.batch_size:
                    batch_inputs = torch.stack(self.input_buffer).to(self.device)
                    batch_targets = torch.stack(self.target_buffer).to(self.device).squeeze()

                    # Training mode
                    self.lnn.train()
                    self.optimizer.zero_grad()
                    outputs = self.lnn(batch_inputs).squeeze()
                    loss = self.loss_function(outputs, batch_targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.lnn.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    # Logging
                    self.logger.info(f"Training Loss: {loss.item()}")
                    for output in outputs:
                        indicator_value = output.item()
                        self.logger.info(f"New Indicator: {indicator_value}")

                    # Clear buffers
                    self.input_buffer = []
                    self.target_buffer = []

                self.data_queue.task_done()

            except asyncio.CancelledError:
                self.logger.info("Data Processor task canceled.")
                break
            except Exception as e:
                self.error_handler.handle_error(
                    f"LNN Processing Error: {e}",
                    exc_info=True
                )
# File: data/data_processor.py
import asyncio
from typing import Dict, Any, List
from error_handler import ErrorHandler
import logging
from models.lnn_model import LiquidNeuralNetwork
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Constants for LNN
INPUT_SIZE = 10  # Adjust based on actual feature size
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1  # Single output for the indicator

# Precomputed scaling parameters (placeholder values)
feature_means = np.array([0.0] * INPUT_SIZE)
feature_stds = np.array([1.0] * INPUT_SIZE)

class DataProcessor:
    def __init__(
        self,
        data_queue: asyncio.Queue,
        lnn_output_queue: asyncio.Queue,
        error_handler: ErrorHandler,
        symbols: List[str],
        timeframes: List[str]
    ):
        self.data_queue = data_queue
        self.lnn_output_queue = lnn_output_queue
        self.error_handler = error_handler
        self.symbols = symbols
        self.timeframes = timeframes
        self.logger = logging.getLogger("DataProcessor")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lnn = LiquidNeuralNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(self.device)
        self.batch_size = 16
        self.input_buffer = []
        self.scaler = StandardScaler()
        # Initialize scaler with precomputed parameters
        self.scaler.mean_ = feature_means
        self.scaler.scale_ = feature_stds
        self.scaler.n_features_in_ = INPUT_SIZE

    def preprocess_data(self, data: Dict[str, Any]) -> torch.Tensor:
        channel = data.get('c')
        if not channel:
            self.logger.warning(f"Received data without channel information: {data}")
            return None

        try:
            if channel.startswith("spot@public.kline.v3.api"):
                return self._preprocess_kline_data(data)
            elif channel.startswith("spot@public.deals.v3.api"):
                return self._preprocess_deals_data(data)
            elif channel.startswith("spot@public.increase.depth.v3.api"):
                return self._preprocess_depth_data(data)
            elif channel.startswith("spot@public.bookTicker.v3.api"):
                return self._preprocess_bookTicker_data(data)
            else:
                self.logger.warning(f"Unknown channel: {channel}")
                return None

        except Exception as e:
            self.error_handler.handle_error(f"Error preprocessing data: {e}", exc_info=True, channel=channel)
            return None

    def _preprocess_kline_data(self, data: Dict[str, Any]) -> torch.Tensor:
        kline_data = data.get('d', {}).get('k')
        if kline_data:
            try:
                features = [
                    float(kline_data['o']),  # Opening price
                    float(kline_data['c']),  # Closing price
                    float(kline_data['h']),  # Highest price
                    float(kline_data['l']),  # Lowest price
                    float(kline_data['v']),  # Quantity
                    float(kline_data['a']),  # Volume
                    int(kline_data['t']),    # Start time
                    int(kline_data['T']),    # End time
                    # Add more features as needed...
                ]
                features = features[:INPUT_SIZE] + [0] * (INPUT_SIZE - len(features))
                features = np.array(features)
                features = (features - self.scaler.mean_) / (self.scaler.scale_ + 1e-8)
                return torch.tensor(features, dtype=torch.float32, device=self.device)

            except (KeyError, TypeError) as e:
                self.error_handler.handle_error(f"Error extracting kline features: {e}", exc_info=True, data=data)
                return None
        return None

    def _preprocess_deals_data(self, data: Dict[str, Any]) -> torch.Tensor:
        deals_data = data.get('d', {}).get('deals')
        if deals_data and isinstance(deals_data, list):
            try:
                first_deal = deals_data[0]
                trade_side = int(first_deal['S'])
                # According to documentation, 'S' = 1 for BUY, 'S' = 2 for SELL
                is_buy = 1 if trade_side == 1 else 0
                features = [
                    float(first_deal['p']),   # Price
                    float(first_deal['v']),   # Quantity/Volume
                    int(first_deal['t']),     # Deal Time
                    is_buy,                   # Trade Side: 1 for BUY, 0 for SELL
                    # Add more features as needed...
                ]
                features = features[:INPUT_SIZE] + [0] * (INPUT_SIZE - len(features))
                features = np.array(features)
                features = (features - self.scaler.mean_) / (self.scaler.scale_ + 1e-8)
                return torch.tensor(features, dtype=torch.float32, device=self.device)

            except (KeyError, TypeError, IndexError) as e:
                self.error_handler.handle_error(f"Error extracting deals features: {e}", exc_info=True, data=data)
                return None
        return None

    def _preprocess_depth_data(self, data: Dict[str, Any]) -> torch.Tensor:
        depth_data = data.get('d')
        if depth_data:
            try:
                asks = depth_data.get('asks', [])
                bids = depth_data.get('bids', [])

                num_levels = min(5, len(asks), len(bids))
                features = []
                for i in range(num_levels):
                    features.extend([
                        float(asks[i]['p']), float(asks[i]['v']),
                        float(bids[i]['p']), float(bids[i]['v'])
                    ])

                features = features[:INPUT_SIZE] + [0] * (INPUT_SIZE - len(features))
                features = np.array(features)
                features = (features - self.scaler.mean_) / (self.scaler.scale_ + 1e-8)
                return torch.tensor(features, dtype=torch.float32, device=self.device)

            except (KeyError, TypeError, IndexError) as e:
                self.error_handler.handle_error(f"Error extracting depth features: {e}", exc_info=True, data=data)
                return None
        return None

    def _preprocess_bookTicker_data(self, data: Dict[str, Any]) -> torch.Tensor:
        bookTicker_data = data.get('d')
        if bookTicker_data:
            try:
                features = [
                    float(bookTicker_data['a']),  # Best ask price
                    float(bookTicker_data['A']),  # Best ask quantity
                    float(bookTicker_data['b']),  # Best bid price
                    float(bookTicker_data['B']),  # Best bid quantity
                    # Add more features as needed...
                ]
                features = features[:INPUT_SIZE] + [0] * (INPUT_SIZE - len(features))
                features = np.array(features)
                features = (features - self.scaler.mean_) / (self.scaler.scale_ + 1e-8)
                return torch.tensor(features, dtype=torch.float32, device=self.device)

            except (KeyError, TypeError) as e:
                self.error_handler.handle_error(f"Error extracting bookTicker features: {e}", exc_info=True, data=data)
                return None
        return None

    async def process_data(self, data: Dict[str, Any]):
        try:
            input_tensor = self.preprocess_data(data)
            if input_tensor is None:
                return

            self.input_buffer.append(input_tensor)

            if len(self.input_buffer) >= self.batch_size:
                batch_tensor = torch.stack(self.input_buffer).to(self.device)
                with torch.no_grad():
                    intelligence_outputs = self.lnn(batch_tensor)
                for intelligence_output in intelligence_outputs:
                    await self.handle_lnn_output(intelligence_output)
                self.input_buffer = []

        except Exception as e:
            self.error_handler.handle_error(
                f"Error processing data through LNN: {e}",
                exc_info=True
            )

    async def handle_lnn_output(self, intelligence_output: torch.Tensor):
        try:
            indicator_value = intelligence_output.item()
            await self.lnn_output_queue.put(indicator_value)
            self.logger.debug(f"LNN indicator enqueued for agents: {indicator_value}")
        except Exception as e:
            self.error_handler.handle_error(
                f"Error handling LNN output: {e}",
                exc_info=True
            )

    async def run(self):
        while True:
            try:
                data = await self.data_queue.get()
                await self.process_data(data)
                self.data_queue.task_done()
            except asyncio.CancelledError:
                self.logger.info("DataProcessor task cancelled.")
                break
            except Exception as e:
                self.error_handler.handle_error(
                    f"Error in DataProcessor run loop: {e}",
                    exc_info=True
                )

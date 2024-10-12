# src/real_time_lnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from collections import deque
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import json
import os
import asyncio
from loguru import logger
from .exceptions import DataProcessingError, ModelTrainingError
from typing import Dict, Any, Optional  # Added Optional here

class LiquidTimeCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super(LiquidTimeCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Layers
        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        self.time_decay = nn.Linear(1, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        # Compute time decay
        decay = torch.exp(-torch.abs(self.time_decay(delta_t.unsqueeze(-1))))
        # Update hidden state
        new_hidden = self.activation(
            self.input2hidden(input) + self.hidden2hidden(hidden * decay)
        )
        new_hidden = self.dropout(new_hidden)
        return new_hidden

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2, dropout: float = 0.0):
        super(LiquidNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a list of LiquidTimeCells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(LiquidTimeCell(cell_input_size, hidden_size, dropout))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence: torch.Tensor, delta_t_sequence: torch.Tensor) -> torch.Tensor:
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)

        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_size, device=input_sequence.device) for _ in range(self.num_layers)]

        for t in range(seq_length):
            x = input_sequence[:, t, :]
            delta_t = delta_t_sequence[:, t]

            for i, cell in enumerate(self.cells):
                h[i] = cell(x, h[i], delta_t)
                x = h[i]

        output = self.output_layer(x)
        return output

class RealTimeLNN:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device(f"cuda:{config['gpu']['device_id']}" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model = LiquidNeuralNetwork(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            output_size=config['model']['output_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['regularization']['dropout']
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['model']['learning_rate'],
            weight_decay=config['model']['regularization']['l2']
        )
        self.criterion = nn.MSELoss()
        self.sequence_length = config['model']['sequence_length']
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        self.grad_scaler = GradScaler()
        self.accumulation_steps = config['model']['accumulation_steps']
        self.step_counter = 0
        self.writer = SummaryWriter(log_dir="logs/real_time_lnn")
        self.training = True  # Flag to control training or inference
        self.checkpoint_interval = config['model']['checkpoint_interval']
        self.last_timestamp = None

        self.data_buffers = {
            'aggTrade': deque(maxlen=self.sequence_length),
            'kline': deque(maxlen=self.sequence_length),
            'depth': deque(maxlen=self.sequence_length),
            '24hrTicker': deque(maxlen=self.sequence_length),
            'bookTicker': deque(maxlen=self.sequence_length),
            'other': deque(maxlen=self.sequence_length)
        }

        self.config = config

    def preprocess_data(self, data: dict) -> np.ndarray:
        try:
            event_type = data.get('e', 'other')
            timestamp = int(data.get('T', data.get('E', 0)))
            features = []

            if event_type == 'aggTrade':
                features = [
                    float(data.get('p', 0)), float(data.get('q', 0)), int(data.get('f', 0)),
                    int(data.get('l', 0)), int(data.get('m', 0))
                ]
            elif event_type == 'kline':
                k = data.get('k', {})
                features = [
                    float(k.get('o', 0)), float(k.get('h', 0)), float(k.get('l', 0)), float(k.get('c', 0)),
                    float(k.get('v', 0)), float(k.get('n', 0)), float(k.get('q', 0)),
                    float(k.get('V', 0)), float(k.get('Q', 0))
                ]
            elif event_type == 'depth':
                bids = data.get('b', [])[:5]
                asks = data.get('a', [])[:5]
                features = [float(price) for bid in bids for price in bid] + \
                           [float(price) for ask in asks for price in ask]
            elif event_type == '24hrTicker':
                features = [
                    float(data.get('p', 0)), float(data.get('P', 0)), float(data.get('w', 0)),
                    float(data.get('c', 0)), float(data.get('Q', 0)), float(data.get('o', 0)),
                    float(data.get('h', 0)), float(data.get('l', 0)), float(data.get('v', 0)),
                    float(data.get('q', 0))
                ]
            elif event_type == 'bookTicker':
                features = [
                    float(data.get('b', 0)), float(data.get('B', 0)),
                    float(data.get('a', 0)), float(data.get('A', 0))
                ]
            else:
                for value in data.values():
                    try:
                        if isinstance(value, (int, float, str)) and value != '':
                            features.append(float(value))
                    except ValueError:
                        continue

            dt = (timestamp - self.last_timestamp) / 1000.0 if self.last_timestamp is not None else 0.0
            self.last_timestamp = timestamp

            return np.array(features + [dt], dtype=np.float32)

        except Exception as e:
            logger.exception(f"Error processing data: {e}")
            raise DataProcessingError("Failed to preprocess data.")

    async def process_message(self, message: str):
        try:
            data = json.loads(message)
            event_type = data.get('e', 'other')
            processed_data = self.preprocess_data(data)

            self.data_buffers[event_type].append(processed_data)

            if all(len(buffer) == self.sequence_length for buffer in self.data_buffers.values()):
                await self.train_step()
        except DataProcessingError as e:
            logger.error(f"Data processing error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error in process_message: {e}")

    async def train_step(self):
        try:
            self.model.train()
            combined_data = np.concatenate(
                [np.array(buffer) for buffer in self.data_buffers.values()],
                axis=1
            )

            if not self.scaler_fitted:
                self.scaler.fit(combined_data)
                self.scaler_fitted = True

            input_data = self.scaler.transform(combined_data)
            dt = input_data[:, -1]
            features = input_data[:, :-1]

            input_tensor = torch.FloatTensor(features[:-1]).unsqueeze(0).to(self.device)
            target = torch.FloatTensor(features[-1]).unsqueeze(0).to(self.device)
            dt_tensor = torch.FloatTensor(dt[:-1]).unsqueeze(0).to(self.device)

            # Data augmentation
            if self.training:
                noise = torch.normal(0, 0.01, size=input_tensor.size()).to(self.device)
                input_tensor += noise

            with autocast():
                output = self.model(input_tensor, dt_tensor)
                loss = self.criterion(output, target) / self.accumulation_steps

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.step_counter += 1
            if self.step_counter % self.accumulation_steps == 0:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
                current_loss = loss.item() * self.accumulation_steps
                logger.info(f"Training loss: {current_loss:.6f}")
                self.writer.add_scalar('Loss/train', current_loss, self.step_counter)

            # Save model periodically
            if self.step_counter % self.checkpoint_interval == 0:
                self.save_model(f'models/model_checkpoint_step_{self.step_counter}.pth')

        except Exception as e:
            logger.exception(f"Error during training step: {e}")
            raise ModelTrainingError("Failed during training step.")

    def save_model(self, path: str = 'models/optimized_real_time_lnn_model.pth'):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state': self.scaler.__getstate__(),
                'step_counter': self.step_counter
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.exception(f"Error saving model: {e}")

    def load_model(self, path: str = 'models/optimized_real_time_lnn_model.pth'):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.__setstate__(checkpoint['scaler_state'])
            self.step_counter = checkpoint['step_counter']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.exception(f"Error loading model: {e}")

    async def predict(self, input_sequence: np.ndarray) -> Optional[np.ndarray]:
        try:
            self.model.eval()
            with torch.no_grad():
                input_data = np.array(input_sequence)
                input_data = self.scaler.transform(input_data)
                dt = input_data[:, -1]
                features = input_data[:, :-1]

                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                dt_tensor = torch.FloatTensor(dt).unsqueeze(0).to(self.device)

                output = self.model(input_tensor, dt_tensor)
                return output.cpu().numpy()
        except Exception as e:
            logger.exception(f"Error during prediction: {e}")
            return None

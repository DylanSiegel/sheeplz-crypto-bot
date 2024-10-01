# src/models/model_factory.py

from typing import Dict, Any
from .base_model import BaseModel
from .lstm_model import LSTMModel
from ..utils.config_manager import ConfigManager

class ModelFactory:
    @staticmethod
    def create(config: ConfigManager) -> BaseModel:
        model_type = config.get('model.type')
        model_params = config.get('model.params')

        if model_type == 'LSTM':
            return LSTMModel(**model_params)
        # Add more model types as needed
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

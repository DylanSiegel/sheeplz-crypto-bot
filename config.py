import yaml
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, field_validator, PositiveInt

class Config(BaseModel):
    """Configuration for DataIngestion and indicators."""

    symbol: str = "BTC_USDT"
    interval: str = "Min1"
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "4h"]
    indicators: List[str] = ["price", "volume", "rsi", "macd", "fibonacci"]

    # GMN Parameters
    max_history_length: int = 1000

    # LNN Parameters
    lnn_model_path: str = "models/lnn/lnn_model.pth"
    lnn_hidden_size: int = 64
    lnn_training_epochs: int = 10
    training_history_length: int = 500
    lnn_learning_rate: float = 0.001

    # Agent Parameters
    threshold_buy: float = 0.7
    threshold_sell: float = 0.3

    # Risk Management
    risk_parameters: Dict[str, Any] = {}

    # Trade Execution
    trade_parameters: Dict[str, Any] = {}

    # System
    agent_loop_delay: PositiveInt = 1
    reconnect_delay: PositiveInt = 5
    log_level: str = "INFO"

    # Device
    device: str = "cpu"

    # Private Channels
    private_channels: List[str] = []

    @field_validator('timeframes', mode='before')
    def validate_timeframes(cls, v):
        if isinstance(v, str):
            return [tf.strip() for tf in v.split(',')]
        elif isinstance(v, list):
            return v
        else:
            raise ValueError("timeframes must be a comma-separated string or a list")

    def load_from_yaml(self, config_path: str):
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, "r") as f:
                config_data: Dict[str, Any] = yaml.safe_load(f)
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config file: {e}")
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Union, Literal
import yaml


class Config(BaseModel):
    """Configuration for Data Ingestion and market data settings."""

    symbol: str = "BTC_USDT"
    interval: str = "1m"  # Default interval, but multiple timeframes are supported.
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "4h"]
    indicators: List[str] = ["price", "volume", "rsi", "macd", "fibonacci"]

    # Websocket and Reconnection Settings
    ws_url: str = "wss://wbs.mexc.com/ws"  # Default; can be overridden by environment variable.
    reconnect_delay: float = 5.0        # Base delay for reconnection attempts
    max_reconnect_attempts: int = 10    # Maximum reconnection attempts
    max_reconnect_delay: float = 300.0  # Maximum delay between reconnection attempts
    backoff_factor: float = 2.0          # Exponential backoff factor

    # Batch Processing Parameters
    batch_size: int = 100               # Number of messages per batch
    batch_time_limit: float = 5.0       # Time limit (seconds) for batching

    # Private Channels (Optional)
    private_channels: List[str] = []

    @field_validator('timeframes')
    def validate_timeframes(cls, v):
        """Validates that 'timeframes' is either a comma-separated string or a list."""
        if isinstance(v, str):
            return [tf.strip() for tf in v.split(',')]
        return v

    def load_from_yaml(self, config_path: str) -> None:
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")

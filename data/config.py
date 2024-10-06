from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import yaml

class DataIngestionConfig(BaseModel):
    """Configuration for Data Ingestion and market data settings."""

    symbol: str = "BTC_USDT"
    interval: str = "Min1"
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "4h"]
    indicators: List[str] = ["price", "volume", "rsi", "macd", "fibonacci"]

    # Private Channels (Optional)
    private_channels: List[str] = []

    # Batch Processing Parameters
    batch_size: int = Field(default=100, ge=1, description="Number of messages per batch to process.")
    batch_time_limit: float = Field(default=5.0, ge=0.1, description="Time limit (in seconds) for batching.")
    rate_limit: int = Field(default=100, ge=1, description="Max number of messages to send per second.")
    reconnect_delay: int = Field(default=5, ge=1, description="Delay (in seconds) between reconnect attempts.")
    max_reconnect_attempts: int = Field(default=10, ge=1, description="Maximum number of allowed reconnect attempts.")
    max_reconnect_delay: float = Field(default=300.0, ge=1.0, description="Maximum delay between reconnect attempts.")

    @field_validator('timeframes', mode='before')
    def validate_timeframes(cls, v):
        """Validates that 'timeframes' is either a comma-separated string or a list."""
        if isinstance(v, str):
            return [tf.strip() for tf in v.split(',')]
        elif isinstance(v, list):
            return v
        else:
            raise ValueError("timeframes must be either a comma-separated string or a list.")

    def load_from_yaml(self, config_path: str) -> None:
        """Loads configuration from a YAML file and updates the instance attributes."""
        try:
            with open(config_path, "r") as f:
                config_data: Dict[str, Any] = yaml.safe_load(f)
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file '{config_path}' not found: {e.filename}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing the YAML config file '{config_path}': {e}")

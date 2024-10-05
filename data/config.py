# data/config.py
from typing import List, Union, Dict, Any
from pydantic import BaseModel, validator, PositiveInt

class Config(BaseModel):
    """Configuration for DataIngestion."""
    symbol: str
    timeframes: List[str]
    private_channels: List[Union[str, Dict[str, Any]]] = []
    reconnect_delay: PositiveInt = 5  # Ensures value > 0
    max_reconnect_delay: PositiveInt = 300
    backoff_factor: float = 2.0
    rate_limit: PositiveInt = 100
    processing_queue_size: PositiveInt = 1000
    max_reconnect_attempts: PositiveInt = 10

    @validator('timeframes', pre=True)
    def validate_timeframes(cls, v):
        if isinstance(v, str):
            return [tf.strip() for tf in v.split(',')]
        elif isinstance(v, list):
            return v
        else:
            raise ValueError("timeframes must be a comma-separated string or a list")

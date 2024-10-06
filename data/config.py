from typing import List, Dict, Any, Union, Optional
from pydantic import BaseSettings, validator, PositiveInt, Field

class Config(BaseSettings):
    """Configuration settings for data ingestion and processing."""

    symbol: str = "BTC_USDT"
    timeframes: List[str] = [
        "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"
    ] 
    private_channels: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    indicators: Dict[str, Any] = Field(default_factory=dict)
    reconnect_delay: PositiveInt = 5
    max_reconnect_delay: PositiveInt = 300
    backoff_factor: float = 2.0
    rate_limit: PositiveInt = 90  
    processing_queue_size: PositiveInt = 1000
    max_reconnect_attempts: int = 10
    max_retry_attempts: int = 3
    data_directory: str = "data" 

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

    @validator('timeframes', pre=True)
    def validate_timeframes(cls, v):
        if isinstance(v, str):
            return [tf.strip() for tf in v.split(',')]
        elif isinstance(v, list):
            return v
        else:
            raise ValueError("Timeframes must be a comma-separated string or a list.")

    def get_indicator_param(
        self, indicator_name: str, param_name: str, default_value: Optional[Any] = None
    ) -> Any:
        """Gets a parameter value for a specific indicator from the config."""
        return self.indicators.get(f"{indicator_name}_{param_name}", default_value)
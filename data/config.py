# data/config.py
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, validator, PositiveInt, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    """Configuration for DataIngestion and indicators."""

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    symbol: str = "BTC_USDT"
    timeframes: List[str] = ["1m", "5m"]
    private_channels: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    indicators: Dict[str, Any] = Field(default_factory=dict)
    reconnect_delay: PositiveInt = 5
    max_reconnect_delay: PositiveInt = 300
    backoff_factor: float = 2.0
    rate_limit: PositiveInt = 100
    processing_queue_size: PositiveInt = 1000
    max_reconnect_attempts: int = 10
    max_retry_attempts: int = 3

    @validator('timeframes', pre=True)
    def validate_timeframes(cls, v):
        if isinstance(v, str):
            return [tf.strip() for tf in v.split(',')]
        elif isinstance(v, list):
            return v
        else:
            raise ValueError("timeframes must be a comma-separated string or a list")

    def get_indicator_param(self, indicator_name: str, param_name: str, default_value: Optional[Any] = None) -> Any:
        return self.indicators.get(f"{indicator_name}_{param_name}", default_value)

    def get_rsi_timeperiod(self) -> int:
        return self.get_indicator_param("rsi", "timeperiod", 14)

    def get_macd_fastperiod(self) -> int:
        return self.get_indicator_param("macd", "fastperiod", 12)

    def get_macd_slowperiod(self) -> int:
        return self.get_indicator_param("macd", "slowperiod", 26)

    def get_macd_signalperiod(self) -> int:
        return self.get_indicator_param("macd", "signalperiod", 9)

    def get_fibonacci_lookback(self) -> int:
        return self.get_indicator_param("fibonacci", "lookback", 14)
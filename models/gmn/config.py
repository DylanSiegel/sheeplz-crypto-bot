import os
from pydantic import BaseModel, Field, validator
from typing import List

class CryptoGMNConfig(BaseModel):
    timeframes: List[str] = ["1m", "5m"]
    indicators: List[str] = ["price", "volume", "rsi", "macd", "fibonacci"]
    max_history: int = 1000
    executor_workers: int = 5
    cache_size: int = 100
    db_path: str = "market_data.db"
    performance_threshold: float = 0.1
    
    @validator('timeframes', 'indicators', pre=True)
    def split_comma_separated(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v

def load_config():
    env = os.getenv("ENVIRONMENT", "DEFAULT").upper()
    config_file = f"config_{env.lower()}.ini" if env != "DEFAULT" else "config.ini"
    
    # Here you would typically load from the config file
    # For simplicity, we're using default values
    return CryptoGMNConfig()
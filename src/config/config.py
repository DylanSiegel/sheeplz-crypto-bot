from dataclasses import dataclass
from typing import List

@dataclass
class TradingConfig:
    # Model parameters
    feature_dim: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    
    # Training parameters
    num_episodes: int = 1000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    
    # Risk management parameters
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_leverage: float = 5.0
    min_trade_interval: int = 5
    
    # Feature extraction parameters
    rolling_window_size: int = 100
    technical_indicators: List[str] = None
    
    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = ["rsi", "volatility", "macd"]

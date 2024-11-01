import numpy as np
from collections import deque
from typing import Dict, List
from src.config import TradingConfig
from src.utils.indicators import calculate_rsi, calculate_volatility, calculate_macd

class FeatureExtractor:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.price_history = deque(maxlen=config.rolling_window_size)
        self.volume_history = deque(maxlen=config.rolling_window_size)
        
    def calculate_features(self, market_data: Dict) -> np.ndarray:
        # Update histories
        self.price_history.append(market_data['close_price'])
        self.volume_history.append(market_data['volume'])
        
        # Calculate base features
        features = [
            self._normalize(market_data['close_price'], self.price_history),
            self._normalize(market_data['volume'], self.volume_history),
            market_data['bid_ask_spread'] / market_data['close_price'],
            market_data['funding_rate'],
            market_data['open_interest'] / np.mean(self.volume_history),
            market_data['leverage_ratio'] / self.config.max_leverage,
            market_data['market_depth_ratio'],
            market_data['taker_buy_ratio']
        ]
        
        # Add technical indicators
        if "rsi" in self.config.technical_indicators:
            features.append(calculate_rsi(list(self.price_history)) / 100.0)
        if "volatility" in self.config.technical_indicators:
            features.append(calculate_volatility(list(self.price_history)))
        if "macd" in self.config.technical_indicators:
            features.append(calculate_macd(list(self.price_history)))
            
        return np.clip(np.array(features), -1, 1)
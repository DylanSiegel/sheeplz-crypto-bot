import numpy as np
import pandas as pd
import ta
from typing import List, Union
from dataclasses import dataclass

@dataclass
class IndicatorConfig:
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2
    
class TechnicalIndicators:
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
        
    def calculate_all(self, prices: Union[List[float], np.ndarray]) -> dict:
        """Calculate all technical indicators"""
        df = pd.DataFrame({'close': prices})
        
        indicators = {}
        
        # RSI
        indicators['rsi'] = self.calculate_rsi(df)
        
        # MACD
        macd_data = self.calculate_macd(df)
        indicators.update(macd_data)
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df)
        indicators.update(bb_data)
        
        # Volatility
        indicators['volatility'] = self.calculate_volatility(df)
        
        return indicators
        
    def calculate_rsi(self, df: pd.DataFrame) -> float:
        """Calculate Relative Strength Index"""
        rsi = ta.momentum.RSIIndicator(
            close=df['close'],
            window=self.config.rsi_period
        )
        return rsi.rsi().iloc[-1]
        
    def calculate_macd(self, df: pd.DataFrame) -> dict:
        """Calculate MACD indicator"""
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=self.config.macd_slow,
            window_fast=self.config.macd_fast,
            window_sign=self.config.macd_signal
        )
        return {
            'macd': macd.macd().iloc[-1],
            'macd_signal': macd.macd_signal().iloc[-1],
            'macd_diff': macd.macd_diff().iloc[-1]
        }
        
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> dict:
        """Calculate Bollinger Bands"""
        bb = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.config.bb_period,
            window_dev=self.config.bb_std
        )
        return {
            'bb_high': bb.bollinger_hband().iloc[-1],
            'bb_mid': bb.bollinger_mavg().iloc[-1],
            'bb_low': bb.bollinger_lband().iloc[-1]
        }
        
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calculate price volatility"""
        returns = df['close'].pct_change()
        return returns.std() * np.sqrt(252)  # Annualized volatility
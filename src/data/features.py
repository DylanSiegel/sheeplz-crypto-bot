import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import ta
from dataclasses import dataclass
from numba import jit

@dataclass
class FeatureConfig:
    """Configuration for feature calculation"""
    window_sizes: List[int] = (5, 10, 20, 50, 100)
    rsi_period: int = 14
    bb_period: int = 20
    epsilon: float = 1e-8
    use_ta_lib: bool = True

@jit(nopython=True)
def calculate_returns(prices: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Calculate log returns with Numba acceleration"""
    return np.log(np.maximum(prices[1:] / prices[:-1], epsilon))

class FeatureCalculator:
    """Unified feature calculator for price data"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def calculate_price_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate basic price-based features"""
        features = {}
        
        # Returns and volatility
        closes = df['Close'].values
        features['log_returns'] = np.pad(calculate_returns(closes), (1, 0))
        features['volatility'] = pd.Series(features['log_returns']).rolling(20).std().fillna(0).values
        
        # Price ratios
        features['high_low_ratio'] = (df['High'] / df['Low']).values
        features['close_open_ratio'] = (df['Close'] / df['Open']).values
        
        # Price position
        range_denominator = (df['High'] - df['Low']).values + self.config.epsilon
        features['price_position'] = ((df['Close'] - df['Low']) / range_denominator).values
        
        return features
    
    def calculate_volume_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate volume-based features"""
        features = {}
        
        # Volume momentum
        volumes = df['Volume'].values
        features['volume_momentum'] = np.pad(np.diff(volumes), (1, 0))
        
        # Volume intensity
        price_range = (df['High'] - df['Low']).values
        features['volume_intensity'] = volumes * price_range
        
        # Volume moving averages
        volume_series = pd.Series(volumes)
        for window in self.config.window_sizes:
            features[f'volume_ma_{window}'] = volume_series.rolling(window).mean().fillna(0).values
            features[f'volume_std_{window}'] = volume_series.rolling(window).std().fillna(0).values
        
        return features
    
    def calculate_technical_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate technical indicators"""
        features = {}
        
        if self.config.use_ta_lib:
            # Momentum indicators
            features['rsi'] = ta.momentum.rsi(df['Close'], window=self.config.rsi_period).fillna(0).values
            
            # Trend indicators
            for window in self.config.window_sizes:
                features[f'sma_{window}'] = ta.trend.sma_indicator(df['Close'], window=window).fillna(0).values
                features[f'ema_{window}'] = ta.trend.ema_indicator(df['Close'], window=window).fillna(0).values
            
            # Volatility indicators
            bb_high = ta.volatility.bollinger_hband(df['Close'], window=self.config.bb_period)
            bb_low = ta.volatility.bollinger_lband(df['Close'], window=self.config.bb_period)
            features['bb_width'] = ((bb_high - bb_low) / df['Close']).fillna(0).values
            
            features['atr'] = ta.volatility.average_true_range(
                df['High'], df['Low'], df['Close']
            ).fillna(0).values
        
        return features
    
    def calculate_all_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate all features"""
        features = {}
        
        # Calculate each feature group
        features.update(self.calculate_price_features(df))
        features.update(self.calculate_volume_features(df))
        features.update(self.calculate_technical_features(df))
        
        # Ensure all features are numpy arrays
        for key, value in features.items():
            if isinstance(value, pd.Series):
                features[key] = value.values
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # Create a small sample dataframe to get feature names
        sample_df = pd.DataFrame({
            'Open': [1] * 10,
            'High': [1] * 10,
            'Low': [1] * 10,
            'Close': [1] * 10,
            'Volume': [1] * 10
        })
        
        return list(self.calculate_all_features(sample_df).keys())
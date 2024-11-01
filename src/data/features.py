import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
import talib
from loguru import logger

@dataclass
class FeatureConfig:
    rolling_window_size: int = 100
    rsi_period: int = 14
    volatility_window: int = 20
    num_threads: int = 24  # Optimized for Ryzen 9 7900X
    use_cuda: bool = True
    batch_size: int = 512
    feature_dim: int = 10

class EnhancedFeatureExtractor:
    """Hardware-optimized feature extractor for high-frequency trading"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
        
        # Initialize circular buffers on GPU if available
        self.price_history = torch.zeros(config.rolling_window_size, device=self.device)
        self.volume_history = torch.zeros(config.rolling_window_size, device=self.device)
        self.current_idx = 0
        
        # Initialize thread pool for CPU calculations
        self.thread_pool = ThreadPoolExecutor(max_workers=config.num_threads)
        
        # Pre-allocate buffers for technical indicators
        self.rsi_buffer = torch.zeros(config.rolling_window_size, device=self.device)
        self.volatility_buffer = torch.zeros(config.rolling_window_size, device=self.device)
        
        # CUDA streams for parallel computation
        self.streams = None
        if self.device.type == 'cuda':
            self.streams = [torch.cuda.Stream() for _ in range(3)]
            
    def update_history(self, price: float, volume: float):
        """Update price and volume history with new data"""
        self.price_history[self.current_idx] = price
        self.volume_history[self.current_idx] = volume
        self.current_idx = (self.current_idx + 1) % self.config.rolling_window_size

    @torch.compile
    def calculate_features_gpu(self, market_data: Dict[str, float]) -> torch.Tensor:
        """Calculate features using GPU acceleration"""
        with torch.cuda.amp.autocast():
            # Update histories
            self.update_history(market_data['close_price'], market_data['volume'])
            
            # Calculate basic features in parallel streams
            if self.streams:
                # Stream 1: Price-based features
                with torch.cuda.stream(self.streams[0]):
                    price_features = self._calculate_price_features(market_data)
                
                # Stream 2: Volume-based features
                with torch.cuda.stream(self.streams[1]):
                    volume_features = self._calculate_volume_features(market_data)
                
                # Stream 3: Technical indicators
                with torch.cuda.stream(self.streams[2]):
                    technical_features = self._calculate_technical_features()
                
                # Synchronize streams
                torch.cuda.synchronize()
                
                # Combine features
                features = torch.cat([price_features, volume_features, technical_features])
            else:
                features = self._calculate_features_sequential(market_data)
            
            return torch.clamp(features, -1, 1)

    @torch.compile
    def _calculate_price_features(self, market_data: Dict[str, float]) -> torch.Tensor:
        """Calculate price-related features"""
        normalized_price = self._normalize_tensor(
            torch.tensor([market_data['close_price']], device=self.device),
            self.price_history
        )
        spread_feature = torch.tensor(
            [market_data['bid_ask_spread'] / market_data['close_price']],
            device=self.device
        )
        funding_feature = torch.tensor(
            [market_data['funding_rate']],
            device=self.device
        )
        return torch.cat([normalized_price, spread_feature, funding_feature])

    @torch.compile
    def _calculate_volume_features(self, market_data: Dict[str, float]) -> torch.Tensor:
        """Calculate volume-related features"""
        normalized_volume = self._normalize_tensor(
            torch.tensor([market_data['volume']], device=self.device),
            self.volume_history
        )
        depth_feature = torch.tensor(
            [market_data['market_depth_ratio']],
            device=self.device
        )
        taker_feature = torch.tensor(
            [market_data['taker_buy_ratio']],
            device=self.device
        )
        return torch.cat([normalized_volume, depth_feature, taker_feature])

    @torch.compile
    def _calculate_technical_features(self) -> torch.Tensor:
        """Calculate technical indicators using GPU"""
        # Calculate RSI
        delta = self.price_history.diff()
        gains = torch.where(delta > 0, delta, torch.zeros_like(delta))
        losses = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        avg_gains = gains.rolling(window=self.config.rsi_period).mean()
        avg_losses = losses.rolling(window=self.config.rsi_period).mean()
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate volatility
        returns = self.price_history.diff() / self.price_history[:-1]
        volatility = returns.std()
        
        # Combine technical features
        technical_features = torch.stack([
            rsi[-1] / 100.0,  # Normalize RSI to [0, 1]
            volatility,
            returns[-1]  # Most recent return
        ])
        
        return technical_features

    @torch.compile
    def _normalize_tensor(self, value: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Normalize values using tensor operations"""
        min_val = history.min()
        max_val = history.max()
        range_val = max_val - min_val
        
        # Handle edge case where range is 0
        mask = range_val > 0
        normalized = torch.where(
            mask,
            (value - min_val) / (range_val + 1e-8) * 2 - 1,
            torch.zeros_like(value)
        )
        return normalized

    def calculate_features(self, market_data: Dict[str, float]) -> np.ndarray:
        """Main feature calculation method with hardware optimization"""
        if self.device.type == 'cuda':
            features = self.calculate_features_gpu(market_data)
            return features.cpu().numpy()
        else:
            # Fallback to CPU implementation with parallel processing
            return self._calculate_features_cpu(market_data)

    def _calculate_features_cpu(self, market_data: Dict[str, float]) -> np.ndarray:
        """CPU-based feature calculation with parallel processing"""
        with Parallel(n_jobs=self.config.num_threads) as parallel:
            # Split calculations across threads
            features = parallel(
                delayed(self._calculate_feature_group)(market_data, group_id)
                for group_id in range(3)
            )
        
        return np.concatenate(features)

    def _calculate_feature_group(self, market_data: Dict[str, float], group_id: int) -> np.ndarray:
        """Calculate a group of features in parallel"""
        if group_id == 0:
            # Price-based features
            return self._calculate_price_features_cpu(market_data)
        elif group_id == 1:
            # Volume-based features
            return self._calculate_volume_features_cpu(market_data)
        else:
            # Technical indicators
            return self._calculate_technical_features_cpu()
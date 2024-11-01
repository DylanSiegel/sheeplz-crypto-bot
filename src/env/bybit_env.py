import gymnasium as gym
import numpy as np
from typing import Tuple, Dict
from src.config import TradingConfig
from src.data.processor import DataProcessor
from src.data.features import FeatureExtractor

class BybitFuturesEnv(gym.Env):
    def __init__(self, config: TradingConfig):
        self.config = config
        self.processor = DataProcessor(config)
        self.feature_extractor = FeatureExtractor(config)
        
        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(config.feature_dim,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        
        # Initialize state
        self.current_price = None
        self.account_balance = None
        self.current_position = None
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize with first market data
        market_data = self._fetch_market_data()
        processed_data = self.processor.preprocess_market_data(market_data)
        features = self.feature_extractor.calculate_features(processed_data)
        
        self.current_price = processed_data['close_price']
        self.account_balance = 10000.0  # Initial balance
        self.current_position = 0
        
        return features, {}
        
    def step(self, action):
        # Execute action and get new market data
        reward = self._execute_action(action)
        market_data = self._fetch_market_data()
        processed_data = self.processor.preprocess_market_data(market_data)
        features = self.feature_extractor.calculate_features(processed_data)
        
        # Update state
        self.current_price = processed_data['close_price']
        
        # Check termination conditions
        done = self._check_termination()
        
        return features, reward, done, False, {}
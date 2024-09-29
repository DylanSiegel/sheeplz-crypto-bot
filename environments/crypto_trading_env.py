# File: environments/crypto_trading_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from rewards.reward_functions import RewardFunction

class CryptoTradingEnv(gym.Env):
    """Custom Gym environment for cryptocurrency trading."""

    def __init__(self, data: pd.DataFrame, processed_features: pd.DataFrame, reward_function: RewardFunction, **kwargs):
        super(CryptoTradingEnv, self).__init__()

        self.data = data
        self.processed_features = processed_features
        self.reward_function = reward_function
        self.initial_balance = kwargs.get('initial_balance', 10000)
        self.transaction_fee = kwargs.get('transaction_fee', 0.001)
        self.slippage = kwargs.get('slippage', 0.001)
        self.max_position = kwargs.get('max_position', 1.0)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(processed_features.shape[1] + 2,), dtype=np.float32)

        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.done = False
        return self._get_observation()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done")

        self.current_step += 1
        current_price = self._get_current_price()
        
        # Execute trade
        trade_amount = action * self.max_position
        if trade_amount > 0:  # Buy
            cost = trade_amount * current_price * (1 + self.slippage)
            if cost <= self.balance:
                self.balance -= cost * (1 + self.transaction_fee)
                self.position += trade_amount
        elif trade_amount < 0:  # Sell
            revenue = abs(trade_amount) * current_price * (1 - self.slippage)
            if abs(trade_amount) <= self.position:
                self.balance += revenue * (1 - self.transaction_fee)
                self.position -= abs(trade_amount)

        # Calculate reward
        next_price = self._get_next_price()
        reward = self.reward_function(
            state=self._get_observation(),
            action=action,
            reward=self._calculate_profit(),
            next_state=self._get_next_observation(),
            done=self.done
        )

        # Check if done
        self.done = self.current_step >= len(self.data) - 1

        return self._get_observation(), reward, self.done, self._get_info()

    def _get_observation(self) -> np.ndarray:
        features = self.processed_features.iloc[self.current_step].values
        return np.concatenate([features, [self.balance, self.position]])

    def _get_next_observation(self) -> np.ndarray:
        next_step = min(self.current_step + 1, len(self.data) - 1)
        features = self.processed_features.iloc[next_step].values
        return np.concatenate([features, [self.balance, self.position]])

    def _get_current_price(self) -> float:
        return self.data.iloc[self.current_step]['close']

    def _get_next_price(self) -> float:
        next_step = min(self.current_step + 1, len(self.data) - 1)
        return self.data.iloc[next_step]['close']

    def _calculate_profit(self) -> float:
        current_price = self._get_current_price()
        portfolio_value = self.balance + self.position * current_price
        return (portfolio_value - self.initial_balance) / self.initial_balance

    def _get_info(self) -> Dict[str, Any]:
        return {
            "balance": self.balance,
            "position": self.position,
            "current_price": self._get_current_price(),
            "profit": self._calculate_profit()
        }

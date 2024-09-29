# File: rewards/reward_functions.py

from abc import ABC, abstractmethod
import numpy as np

class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool) -> float:
        pass

class ProfitReward(RewardFunction):
    def __call__(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool) -> float:
        return reward

class SharpeRatioReward(RewardFunction):
    def __init__(self, window_size: int = 20, risk_free_rate: float = 0.0):
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.returns = []

    def __call__(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool) -> float:
        self.returns.append(reward)
        if len(self.returns) < self.window_size:
            return 0.0
        
        returns_array = np.array(self.returns[-self.window_size:])
        excess_returns = returns_array - self.risk_free_rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        return sharpe_ratio


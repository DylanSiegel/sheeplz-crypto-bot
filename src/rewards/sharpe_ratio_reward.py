# File: src/rewards/sharpe_ratio_reward.py

from .base_reward import RewardFunction
import numpy as np

class SharpeRatioReward(RewardFunction):
    def __init__(self, risk_free_rate: float = 0.0, window_size: int = 20):
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.returns = []

    def calculate_reward(self, action: int, current_price: float, next_price: float, portfolio_value: float) -> float:
        return_ = (next_price - current_price) / current_price
        self.returns.append(return_)

        if len(self.returns) < self.window_size:
            return 0.0

        returns_array = np.array(self.returns[-self.window_size:])
        excess_returns = returns_array - self.risk_free_rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

        return sharpe_ratio

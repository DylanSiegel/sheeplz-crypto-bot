import numpy as np
from typing import List

class RewardFunction:
    def calculate_reward(self, action: int, current_price: float, next_price: float, portfolio_value: float) -> float:
        pass

class ProfitReward(RewardFunction):
    def calculate_reward(self, action: int, current_price: float, next_price: float, portfolio_value: float) -> float:
        if action == 1:  # Buy
            return (next_price - current_price) / current_price
        elif action == 2:  # Sell
            return (current_price - next_price) / current_price
        else:  # Hold
            return 0

class SharpeRatioReward(RewardFunction):
    def __init__(self, risk_free_rate: float = 0.0, window_size: int = 20):
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.returns = []

    def calculate_reward(self, action: int, current_price: float, next_price: float, portfolio_value: float) -> float:
        return_ = (next_price - current_price) / current_price
        self.returns.append(return_)

        if len(self.returns) < self.window_size:
            return 0

        returns_array = np.array(self.returns[-self.window_size:])
        excess_returns = returns_array - self.risk_free_rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

        return sharpe_ratio

class CombinedReward(RewardFunction):
    def __init__(self, profit_weight: float = 0.5, sharpe_weight: float = 0.5):
        self.profit_reward = ProfitReward()
        self.sharpe_reward = SharpeRatioReward()
        self.profit_weight = profit_weight
        self.sharpe_weight = sharpe_weight

    def calculate_reward(self, action: int, current_price: float, next_price: float, portfolio_value: float) -> float:
        profit = self.profit_reward.calculate_reward(action, current_price, next_price, portfolio_value)
        sharpe = self.sharpe_reward.calculate_reward(action, current_price, next_price, portfolio_value)
        return self.profit_weight * profit + self.sharpe_weight * sharpe
# File: src/rewards.py

from abc import ABC, abstractmethod
import numpy as np

class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def calculate_reward(self, action: int, current_price: float, next_price: float, portfolio_value: float) -> float:
        """
        Calculates the reward for a given action and market state.
        """
        pass

class ProfitReward(RewardFunction):
    """Reward function based on realized profit."""

    def calculate_reward(self, action: int, current_price: float, next_price: float, portfolio_value: float) -> float:
        if action == 1:  # Buy
            return (next_price - current_price) / current_price
        elif action == 2:  # Sell
            return (current_price - next_price) / current_price
        else:  # Hold
            return 0

class SharpeRatioReward(RewardFunction):
    """Reward function based on Sharpe ratio."""

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.returns = []

    def calculate_reward(self, action: int, current_price: float, next_price: float, portfolio_value: float) -> float:
        # Calculate portfolio return
        return_ = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value if hasattr(self, 'prev_portfolio_value') else 0
        self.returns.append(return_)
        self.prev_portfolio_value = portfolio_value

        # Calculate Sharpe ratio (if enough data is available)
        if len(self.returns) > 2:
            sharpe_ratio = (np.mean(self.returns) - self.risk_free_rate) / np.std(self.returns)
            return sharpe_ratio
        else:
            return 0

# Example usage
# reward_function = ProfitReward()
# reward = reward_function.calculate_reward(action=1, current_price=100, next_price=110, portfolio_value=1000)
# File: src/rewards/profit_reward.py

from .base_reward import RewardFunction

class ProfitReward(RewardFunction):
    def calculate_reward(self, action: int, current_price: float, next_price: float, portfolio_value: float) -> float:
        if action == 1:  # Buy
            return (next_price - current_price) / current_price
        elif action == 2:  # Sell
            return (current_price - next_price) / current_price
        else:  # Hold
            return 0.0

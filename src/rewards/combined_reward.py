# File: src/rewards/combined_reward.py

from .base_reward import RewardFunction
from .profit_reward import ProfitReward
from .sharpe_ratio_reward import SharpeRatioReward

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

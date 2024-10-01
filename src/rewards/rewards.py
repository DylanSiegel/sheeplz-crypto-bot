# File: src/rewards/rewards.py

from .base_reward import RewardFunction
from .profit_reward import ProfitReward
from .sharpe_ratio_reward import SharpeRatioReward
from .combined_reward import CombinedReward  # Ensure CombinedReward is defined in combined_reward.py

def get_reward_function(reward_type: str) -> RewardFunction:
    """
    Factory function to get the appropriate reward function.

    Args:
        reward_type (str): Type of reward function ('profit', 'sharpe_ratio', 'combined').

    Returns:
        RewardFunction: An instance of the requested reward function.
    """
    if reward_type == "profit":
        return ProfitReward()
    elif reward_type == "sharpe_ratio":
        return SharpeRatioReward()
    elif reward_type == "combined":
        return CombinedReward()
    else:
        raise ValueError(f"Unsupported reward type: {reward_type}")

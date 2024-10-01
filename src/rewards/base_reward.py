# File: src/rewards/base_reward.py

from abc import ABC, abstractmethod
import numpy as np

class RewardFunction(ABC):
    @abstractmethod
    def calculate_reward(self, action: int, current_price: float, next_price: float, portfolio_value: float) -> float:
        """
        Calculates the reward based on the action and price changes.

        Args:
            action (int): The action taken (e.g., Buy, Sell, Hold).
            current_price (float): The current price of the asset.
            next_price (float): The next price of the asset.
            portfolio_value (float): The current portfolio value.

        Returns:
            float: The calculated reward.
        """
        pass

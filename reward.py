# File: reward.py

import numpy as np

def calculate_reward(current_price: float, previous_price: float, action: np.ndarray) -> float:
    """
    Calculates the reward based on the change in price and the agent's action.

    Args:
        current_price (float): The current price of the asset.
        previous_price (float): The previous price of the asset.
        action (np.ndarray): The action taken by the agent.

    Returns:
        float: Calculated reward.
    """
    price_change = current_price - previous_price
    # Assuming action[0] controls position size (e.g., buy/sell)
    position = action[0]
    # Reward is profit from position minus penalty for action magnitude
    profit = price_change * position
    penalty = np.linalg.norm(action) * 0.01  # Adjust penalty coefficient as needed
    return profit - penalty

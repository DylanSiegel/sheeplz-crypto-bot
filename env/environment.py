# File: env/environment.py

import numpy as np
from typing import Tuple
from reward import calculate_reward

class HistoricalEnvironment:
    """Simulates stepping through historical data."""
    def __init__(self, historical_data: np.ndarray):
        self.historical_data = historical_data
        self.current_time = 0
        self.max_time = len(historical_data) - 1

    def reset(self) -> np.ndarray:
        """Resets environment to start."""
        self.current_time = 0
        return self.historical_data[self.current_time]

    def step(self, action: np.ndarray, current_time: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Steps to next time, returns next_state, reward, done, info."""
        next_time = current_time + 1
        done = next_time >= self.max_time
        if not done:
            next_state = self.historical_data[next_time]
        else:
            next_state = np.zeros_like(self.historical_data[0])
        
        # Calculate reward using the reward function
        previous_price = self.historical_data[current_time, 3]  # Assuming close price is at index 3
        current_price = self.historical_data[next_time, 3] if not done else previous_price
        reward = calculate_reward(current_price, previous_price, action)
        
        return next_state, reward, done, {}

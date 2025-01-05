# File: env/environment.py

import numpy as np

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

    def step(self, action: np.ndarray, current_time: int):
        """Steps to next time, returns next_state, reward, done, info."""
        next_time = current_time + 1
        done = next_time >= self.max_time
        if not done:
            next_state = self.historical_data[next_time]
        else:
            next_state = np.zeros_like(self.historical_data[0])
        # Simple reward: random small change plus action effect
        reward = float(np.random.randn() * 0.01 + 0.01)
        return next_state, reward, done, {}

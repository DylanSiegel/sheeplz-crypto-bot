# File: environments/crypto_trading_env.py

import gym
from gym import spaces
import numpy as np
from src.data_acquisition import DataProvider
from src.rewards import RewardFunction

class CryptoTradingEnv(gym.Env):
    """
    Custom Gym environment for cryptocurrency trading.
    """

    def __init__(self, data_provider: DataProvider, reward_function: RewardFunction, initial_balance: float = 1000):
        super(CryptoTradingEnv, self).__init__()

        self.data_provider = data_provider
        self.reward_function = reward_function
        self.initial_balance = initial_balance

        # Define action space (0: Hold, 1: Buy, 2: Sell)
        self.action_space = spaces.Discrete(3)

        # Define observation space (price, volume, indicators)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32)  # Adjust shape as needed

        # Load data
        self.data = None  # This will be set in reset()

        # Initialize environment state
        self.reset()

    def reset(self):
        # Reset environment to initial state
        self.balance = self.initial_balance
        self.position = 0  # Number of units held
        self.current_step = 0

        # Load new data for this episode
        self.data = self.data_provider.get_data("BTC/USDT", "1h", "2023-01-01", "2023-01-31")  # Example parameters

        return self._get_observation()

    def step(self, action: int):
        # Execute trading action
        current_price = self._get_current_price()

        if action == 1:  # Buy
            shares_to_buy = self.balance / current_price
            self.position += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2:  # Sell
            self.balance += self.position * current_price
            self.position = 0

        # Move to next time step
        self.current_step += 1

        # Calculate reward
        next_price = self._get_next_price()
        portfolio_value = self.balance + self.position * next_price
        reward = self.reward_function.calculate_reward(action, current_price, next_price, portfolio_value)

        # Check if episode is done
        done = self.current_step >= len(self.data) - 1

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # Construct observation vector
        obs = self.data.iloc[self.current_step]
        return np.array([
            obs['close'],
            obs['volume'],
            obs['SMA_20'],
            obs['RSI'],
            obs['MACD'],
            obs['ATR'],
            obs['pct_change'],
            obs['volatility'],
            self.balance,
            self.position
        ])

    def _get_current_price(self):
        return self.data.iloc[self.current_step]['close']

    def _get_next_price(self):
        return self.data.iloc[self.current_step + 1]['close']

# Example usage
# env = CryptoTradingEnv(data_provider=data_provider, reward_function=reward_function)
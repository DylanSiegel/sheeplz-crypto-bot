# File: environments/crypto_trading_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any
from src.data_acquisition import DataProvider
from src.rewards import RewardFunction

class CryptoTradingEnv(gym.Env):
    """
    Enhanced custom Gym environment for cryptocurrency trading.
    """

    def __init__(self, 
                 data_provider: DataProvider, 
                 reward_function: RewardFunction, 
                 initial_balance: float = 10000,
                 transaction_fee: float = 0.001,
                 slippage: float = 0.001,
                 window_size: int = 100,
                 max_drawdown: float = 0.2):
        super(CryptoTradingEnv, self).__init__()

        self.data_provider = data_provider
        self.reward_function = reward_function
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.window_size = window_size
        self.max_drawdown = max_drawdown

        # Define action space (-1 to 1, representing sell all to buy all)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Define observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        # Load data
        self.data = None
        self.obs_mean = None
        self.obs_std = None

        # Initialize environment state
        self.reset()

    def reset(self):
        # Reset environment to initial state
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.max_balance = self.initial_balance

        # Load new data for this episode
        self.data = self.data_provider.get_data("BTC/USDT", "1h", "2023-01-01", "2023-12-31")
        
        # Calculate observation normalization parameters
        self._calculate_normalization_params()

        return self._get_observation()

    def step(self, action: float):
        # Execute trading action
        current_price = self._get_current_price()
        
        if action > 0:  # Buy
            max_buyable = self.balance / (current_price * (1 + self.slippage))
            shares_to_buy = max_buyable * action
            cost = shares_to_buy * current_price * (1 + self.slippage)
            self.balance -= cost * (1 + self.transaction_fee)
            self.position += shares_to_buy
        elif action < 0:  # Sell
            shares_to_sell = self.position * abs(action)
            revenue = shares_to_sell * current_price * (1 - self.slippage)
            self.balance += revenue * (1 - self.transaction_fee)
            self.position -= shares_to_sell

        # Move to next time step
        self.current_step += 1

        # Calculate reward
        next_price = self._get_next_price()
        portfolio_value = self.balance + self.position * next_price
        reward = self.reward_function.calculate_reward(action, current_price, next_price, portfolio_value)

        # Update max balance for drawdown calculation
        self.max_balance = max(self.max_balance, portfolio_value)

        # Check if episode is done
        done = self._is_done()

        # Get additional info
        info = self._get_info()

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        obs = self.data.iloc[self.current_step]
        raw_obs = np.array([
            obs['open'],
            obs['high'],
            obs['low'],
            obs['close'],
            obs['volume'],
            obs['SMA_20'],
            obs['EMA_50'],
            obs['RSI'],
            obs['MACD'],
            obs['ATR'],
            obs['pct_change'],
            self.balance,
            self.position
        ])
        return (raw_obs - self.obs_mean) / self.obs_std  # Normalize

    def _get_current_price(self):
        return self.data.iloc[self.current_step]['close']

    def _get_next_price(self):
        return self.data.iloc[self.current_step + 1]['close'] if self.current_step + 1 < len(self.data) else self.data.iloc[-1]['close']

    def _calculate_normalization_params(self):
        # Calculate mean and std for normalization
        obs_data = self.data[['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'EMA_50', 'RSI', 'MACD', 'ATR', 'pct_change']]
        self.obs_mean = np.array(obs_data.mean().tolist() + [self.initial_balance, 0])
        self.obs_std = np.array(obs_data.std().tolist() + [self.initial_balance, self.initial_balance / self._get_current_price()])

    def _is_done(self):
        # Check if we've reached the end of the data
        if self.current_step >= len(self.data) - 1:
            return True
        
        # Check for bankruptcy
        if self.balance <= 0 and self.position <= 0:
            return True
        
        # Check for max drawdown
        portfolio_value = self.balance + self.position * self._get_current_price()
        drawdown = (self.max_balance - portfolio_value) / self.max_balance
        if drawdown > self.max_drawdown:
            return True
        
        return False

    def _get_info(self) -> Dict[str, Any]:
        portfolio_value = self.balance + self.position * self._get_current_price()
        return {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position,
            "portfolio_value": portfolio_value,
            "drawdown": (self.max_balance - portfolio_value) / self.max_balance,
            "max_balance": self.max_balance
        }

    def render(self, mode='human'):
        """
        Render the environment to the screen
        """
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance:.2f}')
            print(f'Position: {self.position:.2f}')
            print(f'Current Price: {self._get_current_price():.2f}')
            print(f'Portfolio Value: {self.balance + self.position * self._get_current_price():.2f}')
            print('----------------------------------------')

# Example usage
# data_provider = DataProvider()
# reward_function = RewardFunction()
# env = CryptoTradingEnv(data_provider=data_provider, reward_function=reward_function)
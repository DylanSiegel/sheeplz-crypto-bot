import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from utils.utils import get_logger

logger = get_logger()


class CryptoTradingEnv(gym.Env):
    """
    A cryptocurrency trading environment for OpenAI gym
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000, transaction_fee_percent: float = 0.001):
        super(CryptoTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        
        # Observation space: Normalized prices + current balance + crypto held
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.df.columns) + 2,), dtype=np.float32
        )

    def reset(self) -> np.array:
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.current_step = 0
        return self._next_observation()

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict]:
        self.current_step += 1
        current_price = self._get_current_price()
        
        if action == 0:  # Buy
            max_buy = self.balance / current_price
            buy_amount = max_buy * 0.1  # Buy 10% of max possible
            impacted_price = self._apply_market_impact(action, buy_amount)
            cost = buy_amount * impacted_price * (1 + self.transaction_fee_percent)
            self.balance -= cost
            self.crypto_held += buy_amount
        elif action == 1:  # Sell
            sell_amount = self.crypto_held * 0.1  # Sell 10% of holdings
            impacted_price = self._apply_market_impact(action, sell_amount)
            revenue = sell_amount * impacted_price * (1 - self.transaction_fee_percent)
            self.balance += revenue
            self.crypto_held -= sell_amount
        
        # Hold case: do nothing
        
        done = self.current_step >= len(self.df) - 1
        obs = self._next_observation()
        reward = self._calculate_reward(current_price)
        info = self._get_info()
        
        return obs, reward, done, info

    def _next_observation(self) -> np.array:
        frame = np.array(self.df.iloc[self.current_step])
        normalized_frame = frame / frame.max()  # Simple normalization
        obs = np.append(normalized_frame, [self.balance / self.initial_balance, self.crypto_held])
        return obs

    def _get_current_price(self) -> float:
        return self.df.iloc[self.current_step]['close']

    def _calculate_reward(self, current_price: float) -> float:
        portfolio_value = self.balance + self.crypto_held * current_price
        return (portfolio_value / self.initial_balance) - 1

    def _get_info(self) -> Dict:
        return {
            'current_step': self.current_step,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'portfolio_value': self.balance + self.crypto_held * self._get_current_price()
        }

    def _apply_market_impact(self, action: int, amount: float) -> float:
        current_price = self._get_current_price()
        impact = 0.1 * (amount / self.df['volume'].iloc[self.current_step])
        if action == 0:  # Buy
            return current_price * (1 + impact)
        elif action == 1:  # Sell
            return current_price * (1 - impact)
        return current_price

    def render(self, mode='human'):
        # Implement visualization logic here if needed
        pass
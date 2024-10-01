# File: src/environments/crypto_trading_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from src.utils.utils import get_logger

logger = get_logger(__name__)

class CryptoTradingEnv(gym.Env):
    """
    A cryptocurrency trading environment for OpenAI gym.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, 
                 transaction_fee_percent: float = 0.001):
        """
        Initializes the CryptoTradingEnv.

        Args:
            df (pd.DataFrame): Historical OHLCV data.
            initial_balance (float, optional): Starting balance in USD. Defaults to 10000.0.
            transaction_fee_percent (float, optional): Transaction fee percentage. Defaults to 0.001.
        """
        super(CryptoTradingEnv, self).__init__()
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0.0
        self.max_steps = len(self.df) - 1

        # Define action and observation space
        # Actions: 0 - Hold, 1 - Buy, 2 - Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [open, high, low, close, volume, balance, crypto_held]
        # Normalize OHLCV by dividing by maximum value in dataset
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.df.columns) - 1 + 2,), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """
        Resets the environment to the initial state.

        Returns:
            np.ndarray: Initial observation.
        """
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.current_step = 0
        return self._next_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes a trading action.

        Args:
            action (int): Action to take (0: Hold, 1: Buy, 2: Sell).

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]:
                - observation (np.ndarray): Next observation.
                - reward (float): Reward obtained.
                - done (bool): Whether the episode has ended.
                - info (Dict[str, Any]): Additional information.
        """
        self.current_step += 1
        done = self.current_step >= self.max_steps

        current_price = self._get_current_price()
        current_volume = self.df.at[self.current_step, 'volume']

        if action == 1:  # Buy
            # Calculate maximum affordable amount (10% of balance)
            max_buy = (self.balance * 0.1) / current_price
            if max_buy > 0:
                impacted_price = self._apply_market_impact(action, max_buy, current_volume)
                cost = max_buy * impacted_price * (1 + self.transaction_fee_percent)
                self.balance -= cost
                self.crypto_held += max_buy
                logger.debug(f"Buy: Amount={max_buy}, Cost={cost}, New Balance={self.balance}, Crypto Held={self.crypto_held}")
        elif action == 2:  # Sell
            # Calculate maximum sellable amount (10% of holdings)
            sell_amount = self.crypto_held * 0.1
            if sell_amount > 0:
                impacted_price = self._apply_market_impact(action, sell_amount, current_volume)
                revenue = sell_amount * impacted_price * (1 - self.transaction_fee_percent)
                self.balance += revenue
                self.crypto_held -= sell_amount
                logger.debug(f"Sell: Amount={sell_amount}, Revenue={revenue}, New Balance={self.balance}, Crypto Held={self.crypto_held}")
        else:
            logger.debug("Hold action taken.")

        # Calculate reward
        reward = self._calculate_reward(current_price)
        logger.debug(f"Step {self.current_step}: Reward={reward}")

        # Get next observation
        obs = self._next_observation()

        # Info dictionary
        info = self._get_info(current_price)

        return obs, reward, done, info

    def _next_observation(self) -> np.ndarray:
        """
        Generates the next observation.

        Returns:
            np.ndarray: Normalized feature vector.
        """
        frame = self.df.iloc[self.current_step].drop('timestamp').values
        normalized_frame = frame / self.df.drop('timestamp', axis=1).max().values
        obs = np.concatenate([normalized_frame, 
                              [self.balance / self.initial_balance, 
                               self.crypto_held / self.initial_balance]])
        return obs.astype(np.float32)

    def _get_current_price(self) -> float:
        """
        Retrieves the current closing price.

        Returns:
            float: Current closing price.
        """
        return self.df.at[self.current_step, 'close']

    def _calculate_reward(self, current_price: float) -> float:
        """
        Calculates the reward based on portfolio value change.

        Args:
            current_price (float): Current closing price.

        Returns:
            float: Calculated reward.
        """
        portfolio_value = self.balance + self.crypto_held * current_price
        reward = (portfolio_value - self.initial_balance) / self.initial_balance
        return reward

    def _get_info(self, current_price: float) -> Dict[str, Any]:
        """
        Provides additional information about the current state.

        Args:
            current_price (float): Current closing price.

        Returns:
            Dict[str, Any]: Information dictionary.
        """
        portfolio_value = self.balance + self.crypto_held * current_price
        return {
            'current_step': self.current_step,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'portfolio_value': portfolio_value
        }

    def _apply_market_impact(self, action: int, amount: float, volume: float) -> float:
        """
        Applies market impact to the trade price based on action and amount.

        Args:
            action (int): Action type (1: Buy, 2: Sell).
            amount (float): Amount of cryptocurrency to trade.
            volume (float): Current volume.

        Returns:
            float: Impacted price.
        """
        impact_factor = 0.1 * (amount / volume)
        if action == 1:  # Buy
            impacted_price = self._get_current_price() * (1 + impact_factor)
        elif action == 2:  # Sell
            impacted_price = self._get_current_price() * (1 - impact_factor)
        else:
            impacted_price = self._get_current_price()
        return impacted_price

    def render(self, mode='human'):
        """
        Renders the current state. Can be expanded for visualization.

        Args:
            mode (str, optional): Render mode. Defaults to 'human'.
        """
        portfolio_value = self.balance + self.crypto_held * self._get_current_price()
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Crypto Held: {self.crypto_held:.4f}")
        print(f"Portfolio Value: {portfolio_value:.2f}")

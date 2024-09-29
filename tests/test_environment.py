# File: tests/test_environment.py

import unittest
from unittest.mock import MagicMock
from environments.crypto_trading_env import CryptoTradingEnv
from src.data.data_acquisition import DataProvider
from src.rewards.rewards import RewardFunction
import numpy as np

class TestCryptoTradingEnv(unittest.TestCase):

    def setUp(self):
        # Mock DataProvider and RewardFunction
        self.mock_data_provider = MagicMock(spec=DataProvider)
        self.mock_reward_function = MagicMock(spec=RewardFunction)
        self.mock_data_provider.get_data.return_value = MagicMock(
            iloc=MagicMock(side_effect=[
                pd.Series({'close': 100, 'SMA_20': 95, 'RSI': 30, 'MACD': 1, 'ATR': 0.5, 'pct_change': 0.01, 'volatility': 0.02}),
                pd.Series({'close': 105, 'SMA_20': 96, 'RSI': 32, 'MACD': 1.2, 'ATR': 0.55, 'pct_change': 0.02, 'volatility': 0.025}),
            ]),
            __len__=MagicMock(return_value=2)
        )
        self.env = CryptoTradingEnv(
            data_provider=self.mock_data_provider,
            reward_function=self.mock_reward_function,
            initial_balance=1000
        )

    def test_reset(self):
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape[0], 10)

    def test_step_buy(self):
        self.mock_reward_function.calculate_reward.return_value = 0.05
        self.env.reset()
        observation, reward, done, info = self.env.step(1)  # Buy
        self.assertEqual(observation.shape[0], 10)
        self.assertEqual(reward, 0.05)
        self.assertFalse(done)

    def test_step_sell(self):
        self.mock_reward_function.calculate_reward.return_value = -0.03
        self.env.reset()
        self.env.step(1)  # Buy
        observation, reward, done, info = self.env.step(2)  # Sell
        self.assertEqual(reward, -0.03)
        self.assertTrue(done)

if __name__ == '__main__':
    unittest.main()

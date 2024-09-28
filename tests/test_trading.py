# File: tests/test_trading.py

import unittest
import pandas as pd
from src.trading import TradingExecutor
from src.rewards import ProfitReward

class TestTradingExecutor(unittest.TestCase):

    def setUp(self):
        # Sample data
        data = {
            'close': [100, 105, 102, 106, 107],
            'SMA_20': [95, 96, 97, 98, 99],
            'EMA': [90, 91, 92, 93, 94],
            'RSI': [30, 32, 31, 33, 34],
            'MACD': [1, 1.2, 1.1, 1.3, 1.4],
            'ATR': [0.5, 0.55, 0.6, 0.65, 0.7],
            'pct_change': [0.01, 0.02, -0.01, 0.03, 0.04],
            'volatility': [0.02, 0.025, 0.03, 0.035, 0.04]
        }
        self.df = pd.DataFrame(data)
        self.features = self.df[['SMA_20', 'EMA', 'RSI', 'MACD', 'ATR', 'pct_change', 'volatility']]
        self.target = pd.Series([0, 1, 0, 1, 0])
        self.reward_function = ProfitReward()
        self.executor = TradingExecutor(initial_balance=1000, transaction_fee=0.001, slippage=0.0005)

    def test_execute_backtest(self):
        trade_history = self.executor.execute_backtest(self.df, self.features, self.target, self.reward_function)
        self.assertEqual(len(trade_history), len(self.features) - 1)
        self.assertIn('action', trade_history.columns)
        self.assertIn('balance', trade_history.columns)
        self.assertIn('position', trade_history.columns)
        self.assertIn('portfolio_value', trade_history.columns)
        self.assertIn('reward', trade_history.columns)

    def test_execute_live_trading(self):
        # Mock model's eval and prediction
        class MockModel:
            def eval(self):
                pass
            def __call__(self, x):
                # Always predict 'Hold'
                return torch.tensor([[0.0, 0.0, 1.0]])

        mock_model = MockModel()
        trade_history = self.executor.execute_live_trading(self.df, self.features, mock_model, self.reward_function)
        self.assertEqual(len(trade_history), len(self.features) - 1)
        self.assertTrue(all(trade_history['action'] == 'Hold'))

if __name__ == '__main__':
    unittest.main()

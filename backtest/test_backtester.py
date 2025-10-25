import unittest
from backtest.backtester import Backtester


def make_synthetic_ohlcv(n=120):
    # make a gentle sine-wave price series with volume spikes
    import math
    data = []
    for i in range(n):
        price = 100 + 2 * math.sin(i * 0.1)
        open_p = price + (0.01 if i % 2 == 0 else -0.01)
        close_p = price
        high = max(open_p, close_p) + 0.1
        low = min(open_p, close_p) - 0.1
        volume = 100 if i % 30 != 0 else 2000
        data.append({'open': open_p, 'high': high, 'low': low, 'close': close_p, 'volume': volume})
    return data


class TestBacktester(unittest.TestCase):
    def test_backtest_runs(self):
        ohlcv = make_synthetic_ohlcv(120)
        bt = Backtester(ohlcv, starting_cash=10000, max_order_usd=10.0)
        res = bt.run()
        self.assertIn('pnl', res)
        self.assertIn('trades_count', res)
        # trades_count should be an int, pnl numeric
        self.assertIsInstance(res['trades_count'], int)
        self.assertIsInstance(res['pnl'], float)


if __name__ == '__main__':
    unittest.main()

import unittest
from integrations.indicators_tv import (
    watchtower_signal,
    bot_activity_idiot_light,
    believe_it_meter,
    livermore_3_points,
    auto_fib_levels,
)


class TestIndicators(unittest.TestCase):

    def test_watchtower_buy(self):
        # upward trending prices -> buy
        prices = [100 + i * 0.5 for i in range(50)]
        sig = watchtower_signal(prices)
        self.assertEqual(sig, 'buy')

    def test_watchtower_none(self):
        prices = [100.0] * 30
        sig = watchtower_signal(prices)
        self.assertIsNone(sig)

    def test_bot_activity_spike(self):
        vols = [10] * 30 + [1000]
        sig = bot_activity_idiot_light(vols, window=20, spike_factor=5.0)
        self.assertEqual(sig, 'spike')

    def test_believe_it_meter_range(self):
        prices = [100 + i * 0.2 for i in range(40)]
        score = believe_it_meter(prices)
        self.assertTrue(-1.0 <= score <= 1.0)

    def test_livermore_3_points(self):
        prices = [10, 9, 11]
        sig = livermore_3_points(prices)
        # a=10,b=9,c=11 => not strictly monotonic -> None
        self.assertIsNone(sig)
        prices = [9, 10, 11]
        self.assertEqual(livermore_3_points(prices), 'buy')

    def test_auto_fib_levels(self):
        prices = [1, 2, 3, 4, 5]
        levels = auto_fib_levels(prices, lookback=5)
        self.assertAlmostEqual(levels['high'], 5.0)
        self.assertAlmostEqual(levels['low'], 1.0)
        self.assertIn('0.618', levels)


if __name__ == '__main__':
    unittest.main()

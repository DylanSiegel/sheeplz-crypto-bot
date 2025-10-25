import unittest
from integrations.tradingview_adapter import combine_indicators_to_action, PriceBuffer, VolumeBuffer


class TestAdapter(unittest.TestCase):
    def test_combiner_buy(self):
        # upward trend prices -> expect positive action
        prices = [100 + i * 0.5 for i in range(50)]
        vols = [100] * 50
        action = combine_indicators_to_action(prices, vols)
        self.assertGreater(action, 0)

    def test_combiner_sell(self):
        prices = [100 - i * 0.5 for i in range(50)]
        vols = [100] * 50
        action = combine_indicators_to_action(prices, vols)
        self.assertLess(action, 0)

    def test_price_volume_buffers(self):
        pb = PriceBuffer(size=5)
        vb = VolumeBuffer(size=5)
        for i in range(6):
            pb.add(100 + i)
            vb.add(100 + i * 10)
        self.assertEqual(len(pb.to_list()), 5)
        self.assertEqual(len(vb.to_list()), 5)


if __name__ == '__main__':
    unittest.main()

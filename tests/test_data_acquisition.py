# File: tests/test_data_acquisition.py

import unittest
from unittest.mock import patch
from src.data_acquisition import BinanceDataProvider
import pandas as pd

class TestBinanceDataProvider(unittest.TestCase):

    @patch('ccxt.binance')
    def test_get_data(self, mock_binance):
        # Setup mock response
        mock_exchange = mock_binance.return_value
        mock_exchange.fetch_ohlcv.return_value = [
            [1609459200000, 29000, 29500, 28800, 29400, 350],
            [1609462800000, 29400, 29600, 29300, 29500, 200]
        ]

        provider = BinanceDataProvider(api_key="test_key", api_secret="test_secret")
        df = provider.get_data(symbol="BTC/USDT", timeframe="1h", start_date="2021-01-01", end_date="2021-01-02")

        # Assertions
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.assertEqual(df.iloc[0]['close'], 29400)
        self.assertEqual(df.iloc[1]['volume'], 200)

if __name__ == '__main__':
    unittest.main()

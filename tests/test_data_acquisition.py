import unittest
from unittest.mock import patch, MagicMock
from src.data.data_acquisition import BinanceDataProvider
import pandas as pd
import ccxt  # Import ccxt for type hinting

class TestBinanceDataProvider(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for the entire test suite (run once)."""
        cls.provider = BinanceDataProvider(api_key="test_key", api_secret="test_secret")

    @patch('ccxt.binance')
    def test_get_data_success(self, mock_binance):
        """Test successful data retrieval."""

        # Set up mock response
        mock_exchange = mock_binance.return_value
        mock_exchange.fetch_ohlcv.return_value = [
            [1609459200000, 29000, 29500, 28800, 29400, 350],
            [1609462800000, 29400, 29600, 29300, 29500, 200]
        ]

        # Call the method
        df = self.provider.get_data(symbol="BTC/USDT", timeframe="1h", 
                                start_date="2021-01-01", end_date="2021-01-02")

        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.assertEqual(df.iloc[0]['close'], 29400)
        self.assertEqual(df.iloc[1]['volume'], 200)
        mock_exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", "1h", 1609459200000, limit=1000)

    @patch('ccxt.binance')
    def test_get_data_api_error(self, mock_binance):
        """Test handling of API errors."""
        mock_exchange = mock_binance.return_value
        mock_exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("Mock Network Error")

        df = self.provider.get_data(symbol="BTC/USDT", timeframe="1h", 
                                start_date="2021-01-01", end_date="2021-01-02")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)  # Expect an empty DataFrame on error

    @patch('ccxt.binance')
    def test_get_multiple_symbols(self, mock_binance):
        """Test retrieving data for multiple symbols."""
        mock_exchange = mock_binance.return_value
        mock_exchange.fetch_ohlcv.side_effect = [
            [[1609459200000, 29000, 29500, 28800, 29400, 350]],  # BTC/USDT
            [[1609459200000, 1300, 1350, 1280, 1340, 500]]    # ETH/USDT
        ]

        symbols = ["BTC/USDT", "ETH/USDT"]
        data = self.provider.get_multiple_symbols(symbols, "1h", "2021-01-01", "2021-01-02")

        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 2)
        self.assertIn("BTC/USDT", data)
        self.assertIn("ETH/USDT", data)
        self.assertIsInstance(data["BTC/USDT"], pd.DataFrame)
        self.assertIsInstance(data["ETH/USDT"], pd.DataFrame)

    @patch('ccxt.binance')
    def test_get_account_balance_success(self, mock_binance):
        """Test successful account balance retrieval."""
        mock_exchange = mock_binance.return_value
        mock_exchange.fetch_balance.return_value = {
            'BTC': {'free': 1.0, 'used': 0.5, 'total': 1.5},
            'USDT': {'free': 1000, 'used': 500, 'total': 1500},
            'ETH': {'free': 0, 'used': 0, 'total': 0} # Test zero balance
        }
        balance = self.provider.get_account_balance()
        self.assertEqual(balance, {
            'BTC': {'free': 1.0, 'used': 0.5, 'total': 1.5},
            'USDT': {'free': 1000, 'used': 500, 'total': 1500}
        })

    @patch('ccxt.binance')
    def test_get_account_balance_error(self, mock_binance):
        """Test account balance retrieval error."""
        mock_exchange = mock_binance.return_value
        mock_exchange.fetch_balance.side_effect = ccxt.AuthenticationError("Mock Authentication Error")

        balance = self.provider.get_account_balance()
        self.assertEqual(balance, {})

    @patch('ccxt.binance')
    def test_place_order_market(self, mock_binance):
        """Test placing a market order."""
        mock_exchange = mock_binance.return_value
        mock_order = {'id': 'mock_order_id'}
        mock_exchange.create_order.return_value = mock_order

        order = self.provider.place_order(symbol="BTC/USDT", order_type="market", side="buy", amount=0.1)
        self.assertEqual(order, mock_order)
        mock_exchange.create_order.assert_called_once_with("BTC/USDT", "market", "buy", 0.1, None)

    @patch('ccxt.binance')
    def test_place_order_limit(self, mock_binance):
        """Test placing a limit order."""
        mock_exchange = mock_binance.return_value
        mock_order = {'id': 'mock_order_id'}
        mock_exchange.create_order.return_value = mock_order

        order = self.provider.place_order(symbol="ETH/USDT", order_type="limit", side="sell", amount=0.05, price=1500)
        self.assertEqual(order, mock_order)
        mock_exchange.create_order.assert_called_once_with("ETH/USDT", "limit", "sell", 0.05, 1500)

    @patch('ccxt.binance')
    def test_place_order_error(self, mock_binance):
        """Test placing an order with an error."""
        mock_exchange = mock_binance.return_value
        mock_exchange.create_order.side_effect = ccxt.InsufficientFunds("Mock Insufficient Funds")

        order = self.provider.place_order(symbol="BTC/USDT", order_type="market", side="buy", amount=100)
        self.assertEqual(order, {})

    def test_rate_limit(self):
        """Test rate limiting logic (rudimentary - you might use more sophisticated mocks for timing)."""
        # ... This would require mocking time.sleep, which is more involved ...
        pass  

if __name__ == '__main__':
    unittest.main()
import unittest
import pandas as pd
from src.data.data_acquisition import BinanceDataProvider 

class TestBinanceDataProvider(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.api_key = 'YOUR_API_KEY'  
        self.api_secret = 'YOUR_API_SECRET' 
        self.data_provider = BinanceDataProvider(self.api_key, self.api_secret)

    def test_get_data(self):
        """Test fetching data from Binance."""
        symbol = 'BTC/USDT'
        timeframe = '1h'
        start_date = '2024-01-01'
        end_date = '2024-01-10'  # Fetch a small amount of data for testing

        df = self.data_provider.get_data(symbol, timeframe, start_date, end_date)

        # Assertions to validate the data
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertGreater(len(df), 0)
        self.assertIn('timestamp', df.columns)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)

    # Add more test methods for other functionalities
    # in BinanceDataProvider (e.g., get_account_balance, place_order).

if __name__ == '__main__':
    unittest.main()
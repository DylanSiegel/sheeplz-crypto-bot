# File: src/data/data_provider.py

from abc import ABC, abstractmethod
import ccxt
import pandas as pd

class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieves historical data for a given symbol and timeframe.
        """
        pass

class BinanceDataProvider(DataProvider):
    """Data provider for Binance exchange."""

    def __init__(self, api_key: str, api_secret: str):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
        })

    def get_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Fetch data from Binance API
        data = self.exchange.fetch_ohlcv(symbol, timeframe, since=start_date, limit=1000)  # Adjust limit as needed

        # Convert to pandas DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df

# Example usage
# data_provider = BinanceDataProvider(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET")
# df = data_provider.get_data(symbol="BTC/USDT", timeframe="1h", start_date="2023-01-01", end_date="2023-01-31")
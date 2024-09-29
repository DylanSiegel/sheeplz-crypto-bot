import ccxt
import pandas as pd
from typing import List
import time
from src.utils.utils import get_logger  # Use absolute import here too

logger = get_logger(__name__)

class BinanceDataProvider:
    def __init__(self, api_key: str, api_secret: str):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        self.rate_limit = 1200  # Binance rate limit (requests per minute)
        self.last_request_time = 0

    def _rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < 60 / self.rate_limit:
            time.sleep((60 / self.rate_limit) - elapsed)
        self.last_request_time = time.time()

    def get_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            self._rate_limit()
            data = self.exchange.fetch_ohlcv(symbol, timeframe, self.exchange.parse8601(start_date), limit=1000)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

    def get_multiple_symbols(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> dict:
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_data(symbol, timeframe, start_date, end_date)
        return data

    def get_account_balance(self) -> dict:
        try:
            self._rate_limit()
            balance = self.exchange.fetch_balance()
            return {asset: balance[asset] for asset in balance if balance[asset]['total'] > 0}
        except Exception as e:
            logger.error(f"Error fetching account balance: {str(e)}")
            return {}

    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> dict:
        try:
            self._rate_limit()
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            return order
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {}
# src/trading/trading_executor.py

import ccxt
from src.utils.utils import get_logger

logger = get_logger(__name__)

class TradingExecutor:
    def __init__(self, api_key: str, api_secret: str, exchange_name: str = 'binance'):
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

    def execute_trade(self, symbol: str, order_type: str, side: str, amount: float, price: float = None):
        """
        Executes a trade on the exchange.

        Args:
            symbol (str): The trading pair (e.g., 'BTC/USDT').
            order_type (str): 'market' or 'limit'.
            side (str): 'buy' or 'sell'.
            amount (float): The amount of cryptocurrency to buy or sell.
            price (float, optional): The limit price for limit orders. Defaults to None.
        """
        try:
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            elif order_type == 'limit':
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            else:
                raise ValueError("Invalid order type. Must be 'market' or 'limit'.")
            
            logger.info(f"Trade executed: {order}")
            return order
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

    # Add more methods for order management, risk management, etc. 
# acquisition/mexc/mexc_data_provider.py

import logging
from typing import List, Dict, Any
from .mexc_rest_api import MexcRestAPI
from .mexc_websocket import MexcWebSocket
from .mexc_order_book import MexcOrderBook
from ..data_provider import DataProvider

class MexcDataProvider(DataProvider):
    """Data provider for MEXC exchange."""

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the MexcDataProvider.

        :param api_key: Your MEXC API key.
        :param api_secret: Your MEXC API secret.
        """
        self.rest_api = MexcRestAPI(api_key, api_secret)
        self.websocket = MexcWebSocket(api_key, api_secret)
        self.order_book = MexcOrderBook()
        self.order_book.set_rest_api(self.rest_api)
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.rest_api.close()

    async def get_historical_data(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """
        Fetch historical kline data.

        :param symbol: Trading pair symbol.
        :param interval: Kline interval.
        :param start_time: Start time in milliseconds.
        :param end_time: End time in milliseconds.
        :return: List of kline data.
        """
        return await self.rest_api.get_klines(symbol, interval, start_time, end_time)

    async def stream_data(self, symbols: List[str], callback: callable):
        """
        Stream real-time kline data for multiple symbols.

        :param symbols: List of trading pair symbols.
        :param callback: Callback function to handle incoming data.
        """
        await self.websocket.subscribe_klines(symbols, callback)

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Fetch account information.

        :return: Dictionary containing account information.
        """
        return await self.rest_api.get_account_info()

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """
        Place an order on the exchange.

        :param symbol: Trading pair symbol.
        :param side: Order side ('BUY' or 'SELL').
        :param order_type: Order type (e.g., 'LIMIT', 'MARKET').
        :param quantity: Order quantity.
        :param price: Order price (required for limit orders).
        :return: Dictionary containing order information.
        """
        return await self.rest_api.place_order(symbol, side, order_type, quantity, price)

    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Fetch the order book for a symbol.

        :param symbol: Trading pair symbol.
        :param limit: Number of price levels to retrieve.
        :return: Dictionary containing order book data.
        """
        return await self.order_book.get_order_book(symbol, limit)

    async def stream_order_book(self, symbol: str, callback: callable):
        """
        Stream real-time order book data for a symbol.

        :param symbol: Trading pair symbol.
        :param callback: Callback function to handle incoming data.
        """
        await self.websocket.subscribe_order_book(symbol, callback)

    async def close(self):
        """Close all connections and resources."""
        await self.rest_api.close()
        # Add any other cleanup operations here if needed
# tests/test_websocket_client.py

import unittest
from unittest.mock import AsyncMock, patch
from src.websocket_client import BinanceFuturesWebSocketClient
from src.exceptions import WebSocketConnectionError

class TestBinanceFuturesWebSocketClient(unittest.IsolatedAsyncioTestCase):
    async def test_connect_success(self):
        symbols = ["btcusdt"]
        on_message = AsyncMock()
        config = {
            'websocket': {
                'base_url': "wss://test.binance.com",
                'ping_interval': 180,
                'pong_timeout': 10,
                'max_retries': 1,
                'backoff_initial': 1,
                'backoff_max': 2
            }
        }

        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            client = BinanceFuturesWebSocketClient(symbols, on_message, config)
            await client.connect()
            self.assertEqual(client.connection_status, 2)  # CONNECTED
            mock_connect.assert_called_once()

    async def test_connect_failure(self):
        symbols = ["btcusdt"]
        on_message = AsyncMock()
        config = {
            'websocket': {
                'base_url': "wss://test.binance.com",
                'ping_interval': 180,
                'pong_timeout': 10,
                'max_retries': 2,
                'backoff_initial': 1,
                'backoff_max': 2
            }
        }

        with patch("websockets.connect", side_effect=Exception("Connection failed")):
            client = BinanceFuturesWebSocketClient(symbols, on_message, config)
            with self.assertRaises(WebSocketConnectionError):
                await client.connect()
            self.assertEqual(client.connection_status, 0)  # DISCONNECTED

if __name__ == '__main__':
    unittest.main()

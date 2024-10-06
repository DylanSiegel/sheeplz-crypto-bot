import os
import json
import asyncio
import websockets
from dotenv import load_dotenv
from typing import List
from error_handler import ErrorHandler

load_dotenv(os.path.join(os.path.dirname(__file__), '../config/.env'))

class MexcWebsocketConnector:
    """
    Connects to the MEXC WebSocket API, subscribes to kline streams, and feeds data into a queue.
    """

    def __init__(self, data_queue: asyncio.Queue, symbols: List[str], timeframes: List[str], error_handler: ErrorHandler):
        """
        Initializes the MexcWebsocketConnector.

        Args:
            data_queue (asyncio.Queue): Queue to put received data for processing.
            symbols (List[str]): List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT']).
            timeframes (List[str]): List of kline timeframes (e.g., ['1m', '15m']).
            error_handler (ErrorHandler): Instance to handle errors.
        """
        self.ws_url = os.getenv("MEXC_WS_URL", "wss://wbs.mexc.com/ws")
        self.api_key = os.getenv("MEXC_API_KEY")
        self.api_secret = os.getenv("MEXC_API_SECRET")
        self.reconnect_delay = 5.0  # Seconds
        self.data_queue = data_queue
        self.symbols = symbols
        self.timeframes = timeframes
        self.max_subscriptions = 30  # As per API limit
        self.error_handler = error_handler

    async def connect(self):
        """
        Establishes WebSocket connection and manages subscription and data reception.
        Reconnects automatically on connection loss.
        """
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    await self._subscribe(ws)
                    # Start keep-alive and data reception concurrently
                    await asyncio.gather(
                        self._receive_data(ws),
                        self._keep_alive(ws)
                    )
            except Exception as e:
                self.error_handler.handle_error(f"WebSocket connection error: {e}", exc_info=True)
                print(f"WebSocket connection error: {e}. Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)

    async def _subscribe(self, ws):
        """
        Subscribes to the specified kline streams.

        Args:
            ws (websockets.WebSocketClientProtocol): Active WebSocket connection.
        """
        subscription_params = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                stream_name = f"spot@public.kline.v3.api@{symbol}@kline_{timeframe}"
                subscription_params.append(stream_name)
                # If subscription_params reach max_subscriptions, send subscription
                if len(subscription_params) == self.max_subscriptions:
                    await self._send_subscription(ws, subscription_params)
                    subscription_params = []
        # Subscribe to any remaining streams
        if subscription_params:
            await self._send_subscription(ws, subscription_params)

    async def _send_subscription(self, ws, params: List[str], method: str = "SUBSCRIPTION", request_id: int = 1):
        """
        Sends subscription or unsubscription requests to the WebSocket.

        Args:
            ws (websockets.WebSocketClientProtocol): Active WebSocket connection.
            params (List[str]): List of stream names to subscribe/unsubscribe.
            method (str): "SUBSCRIPTION" or "UNSUBSCRIPTION".
            request_id (int): Identifier for the request.
        """
        subscription_message = {
            "method": method,
            "params": params,
            "id": request_id
        }
        await ws.send(json.dumps(subscription_message))
        print(f"{method.capitalize()} to streams: {params}")

    async def _receive_data(self, ws):
        """
        Receives data from the WebSocket and puts relevant kline data into the queue.

        Args:
            ws (websockets.WebSocketClientProtocol): Active WebSocket connection.
        """
        async for message in ws:
            try:
                data = json.loads(message)
                stream = data.get('stream')
                if not stream:
                    continue  # Ignore non-stream messages

                # Check if the message is a kline update
                if "kline" in stream:
                    await self.data_queue.put(data)
            except json.JSONDecodeError:
                self.error_handler.handle_error(f"Failed to decode message: {message}", symbol=None, timeframe=None)
            except Exception as e:
                self.error_handler.handle_error(f"Error processing message: {e}", exc_info=True, symbol=None, timeframe=None)

    async def _keep_alive(self, ws):
        """
        Sends PING messages periodically to keep the WebSocket connection alive.

        Args:
            ws (websockets.WebSocketClientProtocol): Active WebSocket connection.
        """
        try:
            while True:
                await asyncio.sleep(60)  # Send PING every 60 seconds
                ping_message = {"method": "PING"}
                await ws.send(json.dumps(ping_message))
                print("Sent PING to WebSocket.")
        except Exception as e:
            self.error_handler.handle_error(f"Error sending PING: {e}", exc_info=True, symbol=None, timeframe=None)

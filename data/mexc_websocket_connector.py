# File: data/mexc_websocket_connector.py
import os
import json
import asyncio
import websockets
from dotenv import load_dotenv
from typing import List
from error_handler import ErrorHandler
import logging
from itertools import count

load_dotenv(os.path.join(os.path.dirname(__file__), '../configs/.env'))

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
            timeframes (List[str]): List of kline timeframes in the correct format (e.g., ['Min1', 'Min5']).
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
        self.logger = logging.getLogger("MexcWebsocketConnector")
        self.request_id_counter = count(1)  # Unique request IDs

    async def connect(self):
        """
        Establishes WebSocket connection and manages subscription and data reception.
        Reconnects automatically on connection loss.
        """
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.logger.info("WebSocket connection established.")
                    await self._subscribe(ws)
                    # Start receiving data
                    await self._receive_data(ws)
            except asyncio.CancelledError:
                self.logger.info("WebSocketConnector task cancelled.")
                break
            except Exception as e:
                self.error_handler.handle_error(f"WebSocket connection error: {e}", exc_info=True)
                self.logger.error(f"WebSocket connection error: {e}. Reconnecting in {self.reconnect_delay} seconds...")
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
                stream_name = f"spot@public.kline.v3.api@{symbol}@{timeframe}"
                subscription_params.append(stream_name)
                # If subscription_params reach max_subscriptions, send subscription
                if len(subscription_params) == self.max_subscriptions:
                    request_id = next(self.request_id_counter)
                    await self._send_subscription(ws, subscription_params, request_id)
                    subscription_params = []
        # Subscribe to any remaining streams
        if subscription_params:
            request_id = next(self.request_id_counter)
            await self._send_subscription(ws, subscription_params, request_id)

    async def _send_subscription(self, ws, params: List[str], request_id: int, method: str = "SUBSCRIPTION"):
        """
        Sends subscription or unsubscription requests to the WebSocket.

        Args:
            ws (websockets.WebSocketClientProtocol): Active WebSocket connection.
            params (List[str]): List of stream names to subscribe/unsubscribe.
            request_id (int): Unique identifier for the request.
            method (str): "SUBSCRIPTION" or "UNSUBSCRIPTION".
        """
        subscription_message = {
            "method": method,
            "params": params,
            "id": request_id
        }
        await ws.send(json.dumps(subscription_message))
        self.logger.info(f"{method.capitalize()} request sent. ID: {request_id}, Streams: {params}")

    async def _receive_data(self, ws):
        """
        Receives data from the WebSocket and puts relevant kline data into the queue.

        Args:
            ws (websockets.WebSocketClientProtocol): Active WebSocket connection.
        """
        async for message in ws:
            try:
                data = json.loads(message)
                
                # Handle PING from server
                if data.get('method') == 'PING':
                    pong_message = {"method": "PONG"}
                    await ws.send(json.dumps(pong_message))
                    self.logger.debug("Received PING, sent PONG.")
                    continue
                
                channel = data.get('c')  # 'c' field contains the channel name
                if not channel:
                    self.logger.debug(f"Ignored message without channel: {data}")
                    continue  # Ignore non-channel messages

                if "kline" in channel:
                    await self.data_queue.put(data)
                    self.logger.debug(f"Received kline data: {data}")
            except json.JSONDecodeError:
                self.error_handler.handle_error(f"Failed to decode message: {message}", symbol=None, timeframe=None)
                self.logger.error(f"Failed to decode message: {message}")
            except Exception as e:
                self.error_handler.handle_error(f"Error processing message: {e}", exc_info=True, symbol=None, timeframe=None)
                self.logger.error(f"Error processing message: {e}")

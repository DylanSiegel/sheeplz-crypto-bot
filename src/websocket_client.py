# src/websocket_client.py

import asyncio
import ssl
import json
from typing import Optional, Callable, List, Dict, Any
import websockets
from loguru import logger
from .exceptions import WebSocketConnectionError

class ConnectionStatus:
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    RECONNECTING = 3

class BinanceFuturesWebSocketClient:
    def __init__(
        self,
        symbols: List[str],
        on_message: Callable[[str], asyncio.Future],
        config: Dict[str, Any]
    ):
        self.base_url = config['websocket']['base_url']
        self.symbols = [s.lower() for s in symbols]
        self.streams = self._get_streams()
        self.connection_status = ConnectionStatus.DISCONNECTED
        self._running = False
        self.on_message = on_message
        self.websocket = None
        self.ping_interval = config['websocket']['ping_interval']
        self.pong_timeout = config['websocket']['pong_timeout']
        self.max_retries = config['websocket']['max_retries']
        self.backoff_initial = config['websocket']['backoff_initial']
        self.backoff_max = config['websocket']['backoff_max']
        self.config = config

    def _get_streams(self) -> List[str]:
        streams = []
        for symbol in self.symbols:
            streams.extend([
                f"{symbol}@aggTrade",
                f"{symbol}@markPrice@1s",
                f"{symbol}@kline_1m",
                f"{symbol}@kline_5m",
                f"{symbol}@kline_15m",
                f"{symbol}@kline_1h",
                f"{symbol}@miniTicker",
                f"{symbol}@ticker",
                f"{symbol}@bookTicker",
                f"{symbol}@forceOrder",
                f"{symbol}@depth20@100ms",
            ])
        streams.extend([
            "!markPrice@arr@1s",
            "!miniTicker@arr",
            "!ticker@arr",
            "!bookTicker",
            "!forceOrder@arr",
        ])
        return streams

    async def connect(self):
        self.connection_status = ConnectionStatus.CONNECTING
        uri = f"{self.base_url}/stream?streams={'/'.join(self.streams)}"

        retry_attempts = 0
        backoff_time = self.backoff_initial

        while retry_attempts < self.max_retries:
            try:
                ssl_context = ssl.create_default_context()
                self.websocket = await websockets.connect(
                    uri,
                    ssl=ssl_context,
                    max_size=2**23,
                    max_queue=1000,
                    compression=None,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.pong_timeout
                )
                logger.info("WebSocket connection established!")
                self.connection_status = ConnectionStatus.CONNECTED
                self._running = True
                return
            except Exception as e:
                retry_attempts += 1
                logger.error(f"Connection attempt {retry_attempts} failed: {e}")
                self.connection_status = ConnectionStatus.DISCONNECTED
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, self.backoff_max)

        logger.critical("Maximum retry attempts reached. Could not establish connection.")
        raise WebSocketConnectionError("Failed to connect to WebSocket after multiple attempts.")

    async def process_message(self, message: str):
        try:
            await self.on_message(message)
        except Exception as e:
            logger.exception(f"An unexpected error occurred during message processing: {e}")

    async def reconnect(self):
        self.connection_status = ConnectionStatus.RECONNECTING
        try:
            await self.close()
            await self.connect()
        except Exception as e:
            logger.error(f"Failed to reconnect: {e}")
            self.connection_status = ConnectionStatus.DISCONNECTED

    async def close(self):
        self._running = False
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("WebSocket connection closed.")
            except Exception as e:
                logger.error(f"Error occurred while closing the WebSocket: {e}")
        self.connection_status = ConnectionStatus.DISCONNECTED

    async def run(self):
        try:
            await self.connect()
            while self._running:
                try:
                    message = await self.websocket.recv()
                    await self.process_message(message)
                except websockets.ConnectionClosed:
                    logger.warning("WebSocket connection closed unexpectedly. Reconnecting...")
                    await self.reconnect()
                except Exception as e:
                    logger.exception(f"Error during message reception: {e}")
                    await self.reconnect()
        except WebSocketConnectionError as e:
            logger.critical(f"WebSocket connection failed: {e}")
        finally:
            await self.close()

    async def subscribe(self, streams: List[str]):
        if self.websocket and self.websocket.open:
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            }
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to streams: {streams}")

    async def unsubscribe(self, streams: List[str]):
        if self.websocket and self.websocket.open:
            unsubscribe_message = {
                "method": "UNSUBSCRIBE",
                "params": streams,
                "id": 1
            }
            await self.websocket.send(json.dumps(unsubscribe_message))
            logger.info(f"Unsubscribed from streams: {streams}")

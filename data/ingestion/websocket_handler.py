# data/ingestion/websocket_handler.py
import asyncio
import json
import logging
import hmac
import hashlib
import time
from typing import Dict, Any, Optional
import websockets
import backoff  # Import backoff library

class WebSocketHandler:
    """Handles the WebSocket connection and communication."""

    def __init__(
        self, url: str, api_key: Optional[str], api_secret: Optional[str], rate_limit: int
    ):
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit = rate_limit
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._last_sent_time = 0
        self._rate_limit_interval = 1.0 / self.rate_limit

    @backoff.on_exception(
        backoff.expo,
        (
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.ConnectionClosedOK,
            ConnectionResetError,
            OSError,
        ),
        max_tries=5,
        giveup=lambda e: isinstance(e, websockets.exceptions.ConnectionClosedOK),
    )
    async def connect(self):
        """Establishes the WebSocket connection."""
        try:
            self.websocket = await websockets.connect(
                self.url, ping_interval=30, ping_timeout=10
            )
            logging.info(f"Connected to {self.url}")
        except Exception as e:
            logging.exception(f"Failed to connect to WebSocket: {e}")
            raise

    async def send_message(self, message: str):
        """Sends a message through the WebSocket."""
        await self._respect_rate_limit()
        if self.websocket and self.websocket.open:
            await self.websocket.send(message)
            logging.debug(f"Sent message: {message}")
        else:
            logging.warning("WebSocket is not connected.")

    async def receive_message(self) -> Optional[str]:
        """Receives a message from the WebSocket."""
        if self.websocket and self.websocket.open:
            try:
                message = await self.websocket.recv()
                logging.debug(f"Received message: {message}")
                return message
            except Exception as e:
                logging.error(f"Error receiving message: {e}")
                raise
        else:
            logging.warning("WebSocket is not connected.")
            return None

    async def close(self):
        """Closes the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            logging.info("WebSocket connection closed.")

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generates a signature for authentication."""
        sorted_params = sorted(params.items())
        query_string = '&'.join(f"{k}={v}" for k, v in sorted_params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def _respect_rate_limit(self):
        """Ensures that messages are sent respecting the rate limit."""
        now = time.time()
        elapsed = now - self._last_sent_time
        if elapsed < self._rate_limit_interval:
            await asyncio.sleep(self._rate_limit_interval - elapsed)
        self._last_sent_time = time.time()

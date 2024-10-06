import asyncio
import hashlib
import hmac
import time
import logging
from typing import Any, Optional, Dict
import websockets
from websockets.client import WebSocketClientProtocol


class WebSocketManager:
    """Manages WebSocket connections, authentication, and rate limiting."""

    def __init__(
        self, ws_url: str, api_key: str, api_secret: str, rate_limit: int, config: Any
    ):
        self.ws_url = ws_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit = rate_limit
        self.config = config
        self.ws: Optional[WebSocketClientProtocol] = None
        self._lock = asyncio.Lock()
        self._last_message_time = 0
        self._rate_limit_interval = 1.0 / rate_limit if rate_limit > 0 else 0

    async def __aenter__(self):
        self.ws = await websockets.connect(self.ws_url)
        return self.ws

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def send_message(
        self, message: str, ws: Optional[WebSocketClientProtocol] = None
    ):
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_message_time
            if elapsed < self._rate_limit_interval:
                await asyncio.sleep(self._rate_limit_interval - elapsed)
            if ws is None:
                ws = self.ws
            if ws is not None:
                await ws.send(message)
                self._last_message_time = time.time()
            else:
                logging.error("WebSocket is not connected.")

    async def receive_message(
        self, ws: Optional[WebSocketClientProtocol] = None
    ) -> Optional[str]:
        if ws is None:
            ws = self.ws
        if ws is not None:
            try:
                message = await ws.recv()
                return message
            except websockets.exceptions.ConnectionClosed as e:
                logging.error(f"WebSocket connection closed: {e}")
                return None
        else:
            logging.error("WebSocket is not connected.")
            return None

    async def close(self):
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        sorted_params = sorted(params.items())
        encoded_params = '&'.join(f"{k}={v}" for k, v in sorted_params)
        message = encoded_params.encode('utf-8')
        secret = self.api_secret.encode('utf-8')
        signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
        return signature

# acquisition/mexc/mexc_websocket.py

import asyncio
import websockets
import json
import logging
from typing import List
from .utils.mexc_auth import generate_signature
from .utils.mexc_error_handling import handle_websocket_error

class MexcWebSocket:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws_url = "wss://wbs.mexc.com/ws"
        self.logger = logging.getLogger(__name__)

    async def _connect(self):
        return await websockets.connect(self.ws_url)

    async def _subscribe(self, websocket, channel: str):
        await websocket.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": [channel],
            "id": 1
        }))

    async def subscribe_klines(self, symbols: List[str], callback: callable):
        backoff = 1
        while True:
            try:
                async with await self._connect() as websocket:
                    for symbol in symbols:
                        await self._subscribe(websocket, f"{symbol}@kline_1m")
                    backoff = 1  # Reset backoff after successful connection

                    while True:
                        response = await websocket.recv()
                        data = json.loads(response)
                        if 'ping' in data:
                            await websocket.send(json.dumps({'pong': data['ping']}))
                        else:
                            await callback(data)
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {str(e)}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)  # Cap the backoff at 60 seconds

    async def subscribe_order_book(self, symbol: str, callback: callable):
        backoff = 1
        while True:
            try:
                async with await self._connect() as websocket:
                    await self._subscribe(websocket, f"{symbol}@depth20")
                    backoff = 1  # Reset backoff after successful connection
                    
                    while True:
                        response = await websocket.recv()
                        data = json.loads(response)
                        if 'ping' in data:
                            await websocket.send(json.dumps({'pong': data['ping']}))
                        else:
                            await callback(data)
            except Exception as e:
                self.logger.error(f"Error in order book stream: {str(e)}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)  # Cap the backoff at 60 seconds
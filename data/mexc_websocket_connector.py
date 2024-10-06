import os
import json
import websockets
import asyncio
from dotenv import load_dotenv

load_dotenv()

class MexcWebsocketConnector:
    def __init__(self, data_queue):
        self.ws_url = os.getenv("MEXC_WS_URL", "wss://wbs.mexc.com/ws")
        self.api_key = os.getenv("MEXC_API_KEY")
        self.api_secret = os.getenv("MEXC_API_SECRET")
        self.reconnect_delay = 5.0
        self.data_queue = data_queue

    async def connect(self):
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    await self._subscribe(ws)
                    await self._receive_batched_klines(ws)
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                await asyncio.sleep(self.reconnect_delay)

    async def _subscribe(self, ws):
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": ["spot@public.kline.v3.api@BTCUSDT@kline_1m", "spot@public.kline.v3.api@BTCUSDT@kline_5m"],
            "id": 1
        }
        await ws.send(json.dumps(subscribe_message))

    async def _receive_batched_klines(self, ws):
        """Receives and processes kline data in batches."""
        kline_batch = []
        while True:
            try:
                message = await ws.recv()
                data = json.loads(message)
                if "spot@public.kline.v3.api" in data.get("c", ""):
                    kline_batch.append(data)
                else:
                    if kline_batch:
                        await self.data_queue.put(kline_batch)
                        kline_batch = []  # Reset batch
            except Exception as e:
                print(f"Error receiving kline data: {e}")
                break

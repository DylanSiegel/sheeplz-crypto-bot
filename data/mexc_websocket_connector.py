# File: data/mexc_websocket_connector.py
import asyncio
import websockets
import json
from typing import List
from error_handler import ErrorHandler
import logging
import time

class MexcWebsocketConnector:
    def __init__(
        self,
        data_queue: asyncio.Queue,
        symbols: List[str],
        timeframes: List[str],
        error_handler: ErrorHandler
    ):
        self.data_queue = data_queue
        self.symbols = symbols
        self.timeframes = timeframes
        self.error_handler = error_handler
        self.logger = logging.getLogger("MexcWebsocketConnector")
        self.uri = "wss://wbs.mexc.com/ws"
        self.ping_interval = 60

    async def connect(self):
        while True:
            try:
                async with websockets.connect(self.uri, ping_interval=None) as websocket:
                    await self.subscribe_channels(websocket)
                    ping_task = asyncio.create_task(self.send_ping(websocket))
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            if data.get("msg") == "PONG":
                                self.logger.debug("Received PONG from server.")
                                continue
                            await self.data_queue.put(data)
                            self.logger.debug(f"Received data: {data}")
                        except json.JSONDecodeError as e:
                            self.error_handler.handle_error(
                                f"JSON decode error: {e}",
                                exc_info=True
                            )
            except websockets.exceptions.ConnectionClosedError as e:
                self.error_handler.handle_error(
                    f"WebSocket connection closed unexpectedly: {e}",
                    exc_info=True
                )
                self.logger.warning("WebSocket connection closed. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                self.error_handler.handle_error(
                    f"Unexpected error in WebSocket connection: {e}",
                    exc_info=True
                )
                self.logger.error("Unexpected error. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def subscribe_channels(self, websocket):
        subscribe_params = self._generate_subscribe_params()
        max_subscriptions = 30
        for i in range(0, len(subscribe_params), max_subscriptions):
            batch = subscribe_params[i:i + max_subscriptions]
            subscribe_message = {
                "method": "SUBSCRIPTION",
                "params": batch,
                "id": int(time.time())
            }
            await websocket.send(json.dumps(subscribe_message))
            self.logger.info(f"Subscribed to {len(batch)} channels.")

            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                response_data = json.loads(response)
                if response_data.get("code") == 0:
                    self.logger.info(f"Subscription successful: {response_data.get('msg')}")
                else:
                    self.error_handler.handle_error(
                        f"Subscription failed: {response_data}",
                        exc_info=False
                    )
            except asyncio.TimeoutError:
                self.error_handler.handle_error(
                    "Subscription acknowledgment timed out.",
                    exc_info=False
                )

    def _generate_subscribe_params(self) -> List[str]:
        channels = []
        for symbol in self.symbols:
            symbol = symbol.upper()
            channels.append(f"spot@public.increase.depth.v3.api@{symbol}")
            channels.append(f"spot@public.deals.v3.api@{symbol}")
            channels.append(f"spot@public.bookTicker.v3.api@{symbol}")
            for timeframe in self.timeframes:
                channels.append(f"spot@public.kline.v3.api@{symbol}@{timeframe}")
        return channels

    async def send_ping(self, websocket):
        try:
            while True:
                ping_message = {
                    "method": "PING"
                }
                await websocket.send(json.dumps(ping_message))
                self.logger.debug("Sent PING to server.")
                await asyncio.sleep(self.ping_interval)
        except asyncio.CancelledError:
            self.logger.info("Ping task cancelled.")
        except Exception as e:
            self.error_handler.handle_error(
                f"Error in PING/PONG mechanism: {e}",
                exc_info=True
            )
            raise e

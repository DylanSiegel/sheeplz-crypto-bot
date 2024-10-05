import asyncio
import logging
import os
from typing import Any, Optional
from asyncio import Queue, CancelledError
from dotenv import load_dotenv
from .config import Config
from .websocket_handler import WebSocketHandler
from .data_processor import DataProcessor

load_dotenv()

class DataIngestion:
    def __init__(
        self,
        gmn: Any,
        config: Config,
        ws_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        self.gmn = gmn
        self.config = config
        self.ws_url = ws_url or os.getenv('MEXC_WS_URL', 'wss://contract.mexc.com/ws')
        self.api_key = api_key or os.getenv('MEXC_API_KEY')
        self.api_secret = api_secret or os.getenv('MEXC_API_SECRET')
        self.websocket_handler = WebSocketHandler(self.ws_url, self.api_key, self.api_secret, config.rate_limit)
        self.data_processor = DataProcessor(gmn)
        self.processing_queue = Queue(maxsize=config.processing_queue_size)
        self._processing_task: Optional[asyncio.Task] = None

    async def connect(self):
        while True:
            try:
                await self.websocket_handler.connect()
                await self._login()
                await self._subscribe()
                self._processing_task = asyncio.create_task(self._process_queue())
                await self._receive_data_loop()
            except Exception as e:
                logging.error(f"Connection error: {e}")
                await asyncio.sleep(self.config.reconnect_delay)

    async def _login(self):
        response = await self.websocket_handler.login()
        # Process login response

    async def _subscribe(self):
        for interval in self.config.timeframes:
            subscription_message = {
                "method": "sub.kline",
                "param": {"symbol": self.config.symbol, "interval": interval},
                "id": self.config.timeframes.index(interval) + 1
            }
            await self.websocket_handler.send_message(subscription_message)
        
        # Subscribe to private channels if authenticated
        if self.api_key and self.api_secret:
            for channel in self.config.private_channels:
                await self.websocket_handler.send_message({
                    "method": "sub",
                    "param": {
                        "channel": channel,
                        "symbol": self.config.symbol
                    }
                })

    async def _receive_data_loop(self):
        while True:
            try:
                message = await self.websocket_handler.receive_message()
                await self.processing_queue.put(message)
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                break

    async def _process_queue(self):
        while True:
            try:
                data = await self.processing_queue.get()
                await self.data_processor.process_data(data)
            except CancelledError:
                break
            except Exception as e:
                logging.error(f"Error processing data: {e}")
            finally:
                self.processing_queue.task_done()

    async def close(self):
        if self._processing_task:
            self._processing_task.cancel()
        await self.websocket_handler.close()
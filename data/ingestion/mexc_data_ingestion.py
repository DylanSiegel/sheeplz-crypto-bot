# data/ingestion/mexc_data_ingestion.py
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List
from asyncio import Queue
from contextlib import suppress
import websockets
from dotenv import load_dotenv

from ..config import Config
from .websocket_handler import WebSocketHandler
from ..processing.data_processor import DataProcessor

load_dotenv()


class DataIngestion:
    """Handles the ingestion of data from the MEXC WebSocket API."""

    def __init__(self, gmn: Any, config: Config):
        """
        Initializes the DataIngestion instance.

        :param gmn: The GMN instance for updating graphs.
        :param config: Configuration object.
        """
        self.gmn = gmn
        self.config = config
        self.ws_url = os.getenv('MEXC_WS_URL', 'wss://contract.mexc.com/ws')
        self.api_key = os.getenv('MEXC_API_KEY')
        self.api_secret = os.getenv('MEXC_API_SECRET')
        self.websocket_handler = WebSocketHandler(
            self.ws_url, self.api_key, self.api_secret, config.rate_limit
        )
        self.data_processor = DataProcessor(gmn)
        self.processing_queue: Queue[Dict[str, Any]] = asyncio.Queue(
            maxsize=config.processing_queue_size
        )

        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.max_reconnect_attempts
        self.backoff_factor = config.backoff_factor
        self._batch_task: Optional[asyncio.Task] = None

    async def connect(self):
        """Connects to the WebSocket and handles reconnections."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                await self.websocket_handler.connect()
                self.reconnect_attempts = 0  # Reset attempts on successful connection
                await self._login()
                await self._subscribe()

                # Start the batch processor task here:
                self._batch_task = asyncio.create_task(self._batch_processor())

                await self._receive_data_loop()  # Start receiving data

            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                ConnectionResetError,
                OSError,
                asyncio.TimeoutError,
            ) as e:
                logging.error(f"WebSocket connection error: {e}")

            except Exception as e:
                logging.exception(f"An unexpected error occurred: {e}")

            finally:
                if (
                    self.websocket_handler.websocket
                    and self.websocket_handler.websocket.open
                ):
                    await self.websocket_handler.close()

            self.reconnect_attempts += 1
            wait_time = min(
                self.config.reconnect_delay
                * (self.backoff_factor ** (self.reconnect_attempts - 1)),
                self.config.max_reconnect_delay,
            )
            logging.info(
                f"Reconnecting attempt {self.reconnect_attempts} in {wait_time:.2f} seconds..."
            )
            await asyncio.sleep(wait_time)

    async def _login(self):
        """Handles the authentication with the WebSocket server."""
        if self.api_key and self.api_secret:
            timestamp = int(time.time())
            sign_params = {
                'api_key': self.api_key,
                'req_time': timestamp,
            }
            signature = self.websocket_handler._generate_signature(sign_params)
            login_params = {
                'method': 'server.auth',
                'params': [self.api_key, timestamp, signature],
                'id': 1,
            }
            await self.websocket_handler.send_message(json.dumps(login_params))
            response = await self.websocket_handler.receive_message()
            if response:
                response_data = json.loads(response)
                if response_data.get('result') == 'success':
                    logging.info("WebSocket login successful.")
                else:
                    logging.error("WebSocket login failed.")
            else:
                logging.error("No response received during WebSocket login.")

    async def _subscribe(self):
        """Subscribes to the required channels."""
        # Subscribing to public channels
        for timeframe in self.config.timeframes:
            kline_channel = f"sub.kline.{self.config.symbol}.{timeframe}"
            subscribe_message = {
                'method': kline_channel,
                'params': [],
                'id': 1,
            }
            await self.websocket_handler.send_message(json.dumps(subscribe_message))

        # Subscribing to private channels
        for channel in self.config.private_channels:
            subscribe_message = {
                'method': f"sub.{channel}",
                'params': [],
                'id': 1,
            }
            await self.websocket_handler.send_message(json.dumps(subscribe_message))

    async def _receive_data_loop(self):
        """Receives data from the websocket."""
        while True:
            try:
                message = await self.websocket_handler.receive_message()
                if message:
                    await self.processing_queue.put(message)
            except websockets.exceptions.ConnectionClosedOK as e:
                logging.warning(f"WebSocket connection closed gracefully: {e.reason}")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                logging.error(f"WebSocket connection closed with error: {e.reason}")
                break
            except asyncio.CancelledError:
                logging.info("Receive data loop cancelled.")
                break
            except Exception as e:
                logging.exception(f"Error in receive_data_loop: {e}")
                break

    async def _batch_processor(self, batch_size: int = 100, timeout: float = 0.5):
        """Processes data in batches from the queue."""
        try:
            while True:
                batch = []
                try:
                    for _ in range(batch_size):
                        message = await asyncio.wait_for(self.processing_queue.get(), timeout)
                        batch.append(json.loads(message))
                        self.processing_queue.task_done()
                except asyncio.TimeoutError:
                    pass  # Process the collected batch
                if batch:
                    try:
                        await self.data_processor.process_data(batch)
                    except Exception as e:
                        logging.error(f"Error processing batch: {e}")
        except asyncio.CancelledError:
            logging.info("Batch processor task cancelled.")
            raise

    async def close(self):
        """Closes the WebSocket connection and tasks."""
        tasks = [self._batch_task]
        for task in tasks:
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        if self.websocket_handler:
            await self.websocket_handler.close()

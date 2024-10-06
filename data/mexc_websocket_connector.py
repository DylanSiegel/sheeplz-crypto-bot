import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List
from asyncio import Queue
from dotenv import load_dotenv
import websockets
import requests

from .config import Config
from .websocket_manager import WebSocketManager  # Corrected import

load_dotenv()

RECONNECT_JITTER = 0.2  # 20% jitter


class MexcRestAPI:
    """Handles REST API interactions with MEXC for listenKey management."""

    def __init__(self, api_key: str, api_secret: str):
        self.base_url = "https://api.mexc.com"
        self.api_key = api_key
        self.api_secret = api_secret

    def get_listen_key(self) -> str:
        """Fetch the listenKey from MEXC API to start the WebSocket connection."""
        url = f"{self.base_url}/api/v3/userDataStream"
        headers = {
            'X-MEXC-APIKEY': self.api_key,
        }
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            listen_key = response.json().get('listenKey')
            if listen_key:
                logging.info("Successfully obtained listenKey.")
                return listen_key
            else:
                raise Exception("Failed to obtain listenKey: No listenKey in response.")
        else:
            raise Exception(f"Failed to get listenKey: {response.text}")

    def refresh_listen_key(self, listen_key: str):
        """Refresh the listenKey before it expires (every 60 minutes)."""
        url = f"{self.base_url}/api/v3/userDataStream"
        headers = {
            'X-MEXC-APIKEY': self.api_key,
        }
        params = {"listenKey": listen_key}
        response = requests.put(url, headers=headers, params=params)
        if response.status_code == 200:
            logging.info("Successfully refreshed listenKey.")
        else:
            raise Exception(f"Failed to refresh listenKey: {response.text}")


class MexcWebsocketConnector:
    """
    Connects to the MEXC WebSocket API, handles subscriptions,
    and passes raw data to a processing queue.
    """

    def __init__(self, config: Config, data_queue: Queue):
        self.config = config
        self.data_queue = data_queue
        self.ws_url = os.getenv("MEXC_WS_URL", "wss://wbs.mexc.com/ws")
        self.api_key = os.getenv("MEXC_API_KEY")
        self.api_secret = os.getenv("MEXC_API_SECRET")
        self.websocket_manager = WebSocketManager(
            self.ws_url, self.api_key, self.api_secret, config.rate_limit, config=config
        )

        self.reconnect_attempts = 0
        self._last_reconnect_time = 0
        self.connected_event = asyncio.Event()

        # Initialize REST API client
        self.rest_api = MexcRestAPI(self.api_key, self.api_secret)
        self.listen_key = None
        self.listen_key_task = None

    async def connect(self) -> None:
        """Connects to the WebSocket and handles reconnections with backoff."""
        try:
            # Get the initial listenKey
            self.listen_key = self.rest_api.get_listen_key()

            # Start the listenKey refresh task
            self.listen_key_task = asyncio.create_task(self._refresh_listen_key_periodically())

            while self.reconnect_attempts < self.config.max_reconnect_attempts:
                try:
                    await self._connect_and_receive()
                except websockets.exceptions.ConnectionClosedError as e:
                    logging.error(f"WebSocket connection error: {e}")
                except Exception as e:
                    logging.exception(f"An unexpected error occurred: {e}")
                finally:
                    self.connected_event.clear()

                await self._handle_reconnection_delay()

        except Exception as e:
            logging.exception(f"Failed to establish WebSocket connection: {e}")

    async def _connect_and_receive(self):
        """Handles the WebSocket connection and data reception."""
        connection_url = f"{self.ws_url}?listenKey={self.listen_key}"
        async with websockets.connect(connection_url, ping_interval=20, ping_timeout=20) as ws:
            self.connected_event.set()
            logging.info("WebSocket connection established.")

            # Subscribe to multiple timeframes
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [f"spot@public.kline.v3.api@BTCUSDT@{tf}" for tf in self.config.timeframes],
                "id": 1
            }
            await self.websocket_manager.send_message(json.dumps(subscribe_message), ws)
            logging.info(f"Subscribed to timeframes: {self.config.timeframes}")

            # Start receiving data
            await self._receive_data_loop(ws)

    async def _receive_data_loop(self, ws) -> None:
        """Receives data from WebSocket and processes it in batches."""
        try:
            data_batch = []
            batch_size = self.config.batch_size  # Ensure Config has batch_size
            batch_time_limit = self.config.batch_time_limit  # Ensure Config has batch_time_limit
            batch_start_time = time.time()

            while True:
                try:
                    message = await ws.recv()
                    data = json.loads(message)
                    data_batch.append(data)

                    # Check if batch size or time limit is reached
                    current_time = time.time()
                    if len(data_batch) >= batch_size or (current_time - batch_start_time) >= batch_time_limit:
                        if data_batch:
                            await self.data_queue.put(data_batch.copy())
                            logging.debug(f"Batch of {len(data_batch)} messages put into queue.")
                            data_batch.clear()
                            batch_start_time = current_time  # Reset timer after each batch

                except asyncio.CancelledError:
                    logging.info("Receive data loop cancelled.")
                    break
                except websockets.exceptions.ConnectionClosedOK as e:
                    logging.warning(f"WebSocket closed gracefully: {e.reason}")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    logging.error(f"WebSocket closed with error: {e.reason}")
                    break
                except Exception as e:
                    logging.exception(f"Error in receive_data_loop: {e}")
                    break

            # Put remaining data in queue
            if data_batch:
                await self.data_queue.put(data_batch.copy())
                logging.debug(f"Final batch of {len(data_batch)} messages put into queue.")

        except Exception as e:
            logging.exception(f"Exception in _receive_data_loop: {e}")

    async def _refresh_listen_key_periodically(self):
        """Periodically refreshes the listenKey before it expires."""
        try:
            while True:
                await asyncio.sleep(30 * 60)  # Refresh every 30 minutes
                try:
                    self.rest_api.refresh_listen_key(self.listen_key)
                except Exception as e:
                    logging.error(f"Error refreshing listenKey: {e}")
        except asyncio.CancelledError:
            logging.info("ListenKey refresh task cancelled.")

    async def _handle_reconnection_delay(self) -> None:
        """Handles delay between reconnection attempts with exponential backoff and jitter."""
        current_time = time.monotonic()
        backoff_delay = self.config.reconnect_delay * (
            self.config.backoff_factor ** self.reconnect_attempts
        )
        jitter = backoff_delay * RECONNECT_JITTER
        wait_time = min(backoff_delay + jitter, self.config.max_reconnect_delay)
        next_reconnect = self._last_reconnect_time + wait_time
        sleep_duration = max(0, next_reconnect - current_time)
        logging.info(
            f"Next reconnection attempt in {sleep_duration:.2f} seconds..."
        )
        await asyncio.sleep(sleep_duration)
        self._last_reconnect_time = time.monotonic()
        self.reconnect_attempts += 1

    async def close(self) -> None:
        """Closes the WebSocket connection and cancels background tasks."""
        if self.listen_key_task:
            self.listen_key_task.cancel()
            try:
                await self.listen_key_task
            except asyncio.CancelledError:
                logging.info("ListenKey refresh task cancelled.")

        await self.websocket_manager.close()
        logging.info("WebSocket connection closed.")
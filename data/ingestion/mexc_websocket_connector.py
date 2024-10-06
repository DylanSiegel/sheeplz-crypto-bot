# data/ingestion/mexc_websocket_connector.py
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List
from asyncio import Queue, Event
from contextlib import suppress
import websockets
import uuid
from dotenv import load_dotenv

from ..config import Config
from .websocket_manager import WebSocketManager  # Correct import

load_dotenv()

class MexcWebsocketConnector: 
    """Connects to the MEXC WebSocket API, handles subscriptions, 
       and passes raw data to a processing queue.
    """

    def __init__(self, config: Config, data_queue: asyncio.Queue):
        self.config = config
        self.data_queue = data_queue 

        self.ws_url = os.getenv("MEXC_WS_URL", "wss://wbs.mexc.com/ws")
        self.api_key = os.getenv("MEXC_API_KEY")
        self.api_secret = os.getenv("MEXC_API_SECRET")
        self.websocket_manager = WebSocketManager(
            self.ws_url, self.api_key, self.api_secret, config.rate_limit, config=config
        )  

        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.max_reconnect_attempts
        self.backoff_factor = config.backoff_factor
        self._last_reconnect_time = 0

        self.connected_event = Event()

    async def connect(self):
        """Connects to the WebSocket and handles reconnections with backoff."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                async with self.websocket_manager as ws: 
                    if self.reconnect_attempts > 0:
                        logging.info("Reconnection successful.")
                    self.reconnect_attempts = 0
                    self.connected_event.set() 

                    await self._login(ws)
                    await self._subscribe(ws)
                    await self._receive_data_loop(ws)

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
                self.connected_event.clear()  

            current_time = time.monotonic()
            backoff_delay = self.config.reconnect_delay * (
                self.backoff_factor ** self.reconnect_attempts
            )
            jitter = backoff_delay * 0.2 
            wait_time = min(backoff_delay + jitter, self.config.max_reconnect_delay)
            next_reconnect = self._last_reconnect_time + wait_time
            sleep_duration = max(0, next_reconnect - current_time)
            logging.info(
                f"Next reconnection attempt in {sleep_duration:.2f} seconds..."
            )
            await asyncio.sleep(sleep_duration)
            self._last_reconnect_time = time.monotonic()
            self.reconnect_attempts += 1

    async def _login(self, ws):
        if self.api_key and self.api_secret:
            timestamp = int(time.time())
            sign_params = {'api_key': self.api_key, 'req_time': timestamp}
            signature = self.websocket_manager._generate_signature(sign_params)
            login_params = {
                'method': 'server.auth',
                'params': [self.api_key, timestamp, signature],
                'id': 1,
            }
            await self.websocket_manager.send_message(json.dumps(login_params), ws)
            response = await self.websocket_manager.receive_message(ws)
            if response:
                response_data = json.loads(response)
                if response_data.get('result') == 'success':
                    logging.info("WebSocket login successful.")
                else:
                    logging.error("WebSocket login failed.")
            else:
                logging.error("No response received during WebSocket login.")

    async def _subscribe(self, ws):
        for timeframe in self.config.timeframes:
            kline_channel = f"sub.kline.{self.config.symbol}.{timeframe}"
            subscribe_message = {'method': kline_channel, 'params': [], 'id': 1}
            await self.websocket_manager.send_message(
                json.dumps(subscribe_message), ws
            )

        for channel in self.config.private_channels:
            subscribe_message = {'method': f"sub.{channel}", 'params': [], 'id': 1}
            await self.websocket_manager.send_message(
                json.dumps(subscribe_message), ws
            )

    async def _receive_data_loop(self, ws):
        while True:
            try:
                message = await ws.receive_message()
                if message:
                    try:
                        data = json.loads(message)
                        await self.data_queue.put(data)
                    except json.JSONDecodeError as e:
                        logging.error(
                            f"Failed to decode JSON message: {e}. Message: {message}"
                        )
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

    async def close(self):
        await self.websocket_manager.close()
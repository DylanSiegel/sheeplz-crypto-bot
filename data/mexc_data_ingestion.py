import asyncio
import json
import logging
import websockets
import os
import hmac
import hashlib
import time
from collections import deque
from typing import Dict, List
from dotenv import load_dotenv
from models.utils.config import Config
import websockets.exceptions as ws_exceptions
import multiprocessing

# Load environment variables from .env file
load_dotenv('.env')

# Number of processes for multiprocessing
NUM_PROCESSES = multiprocessing.cpu_count()  # Use all available CPU cores


class DataIngestion:
    """
    Handles the ingestion of real-time kline data from the MEXC websocket API.
    Uses asynchronous operations for non-blocking data reception and processing.
    Employs multiprocessing for parallel calculation of technical indicators.
    Utilizes a deque for buffering kline data.
    Includes data validation and more robust error handling.
    """

    def __init__(self, gmn, config: Config):
        self.config = config
        self.gmn = gmn
        self.symbol = config.symbol.replace("_", "/")  # Convert BTC_USDT to BTC/USDT (if needed)
        self.interval = config.interval
        self.ws_url = os.getenv('MEXC_WS_URL', 'wss://contract.mexc.com/ws')
        self.api_key = os.getenv('MEXC_API_KEY')
        self.api_secret = os.getenv('MEXC_API_SECRET')
        self._kline_buffer = deque(maxlen=config.max_history_length)  # Buffer for kline data
        self.pool = multiprocessing.Pool(processes=NUM_PROCESSES)  # Create a multiprocessing pool

    async def connect(self):
        """
        Establishes a connection to the MEXC websocket API and manages the data reception loop.
        Includes a reconnect mechanism to handle connection interruptions.
        """
        while True:  # Reconnect loop
            try:
                async with websockets.connect(self.ws_url, ping_interval=None) as websocket:
                    logging.info(f"Connected to {self.ws_url}")

                    if self.api_key and self.api_secret:
                        await self._login(websocket)

                    await self._subscribe_public_channels(websocket)
                    await self._receive_data_loop(websocket)

            except (ws_exceptions.ConnectionClosedError, ConnectionError, OSError) as e:
                logging.error(f"WebSocket connection closed with error: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")

            logging.info(f"Reconnecting in {self.config.reconnect_delay} seconds...")
            await asyncio.sleep(self.config.reconnect_delay)

    async def _login(self, websocket):
        """Logs in to the MEXC websocket API using the provided API key and secret."""
        timestamp = int(time.time() * 1000)
        signature = self._generate_signature(timestamp)
        login_message = {
            "method": "login",
            "param": {
                "apiKey": self.api_key,
                "signature": signature,
                "timestamp": timestamp
            }
        }
        await websocket.send(json.dumps(login_message))
        response = await websocket.recv()
        logging.info(f"Login response: {response}")

    def _generate_signature(self, timestamp: int) -> str:
        """Generates the signature required for logging in."""
        message = f"{self.api_key}{timestamp}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _subscribe_public_channels(self, websocket):
        """Subscribes to public kline channels for the specified symbol and timeframes."""
        channels = [
            {"method": "sub.kline", "param": {"symbol": f"{self.symbol}.P", "interval": interval}}
            for interval in self.config.timeframes
        ]
        for channel in channels:
            await websocket.send(json.dumps(channel))
            await asyncio.sleep(0.1)  # Small delay to prevent overloading the server
        logging.info(f"Subscribed to channels: {channels}")

    async def _receive_data_loop(self, websocket):
        """Continuously receives and processes incoming websocket messages."""
        while True:
            try:
                message = await websocket.recv()
                await self._process_message(message)

            except websockets.exceptions.ConnectionClosed:
                logging.error("Websocket connection closed.")
                break
            except Exception as e:
                logging.exception(f"Exception in _receive_data_loop: {e}")
                break

    async def _process_message(self, message):
        """Processes an individual websocket message and validates the data."""
        try:
            data = json.loads(message)

            # Validate if the message is a kline data push
            if 'data' in data and 'channel' in data and data['channel'].startswith(f"{self.symbol}.P@kline"):
                kline_data = data['data']

                # Validate kline_data structure
                if isinstance(kline_data, dict) and all(key in kline_data for key in ('t', 'o', 'h', 'l', 'c', 'v')):
                    self._kline_buffer.append(kline_data)
                    await self.gmn.update_graph([kline_data], self.pool)
                    logging.debug(f"Received kline data: {kline_data}")
                else:
                    logging.warning(f"Invalid kline data format: {kline_data}")

        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON data received: {e}. Message: {message}")
        except Exception as e:
            logging.exception(f"Error processing message: {e}. Message: {message}")

    async def close(self):
        """Closes the websocket connection and terminates the multiprocessing pool."""
        self.pool.close()  # Close the multiprocessing pool
        self.pool.join()  # Wait for all processes to finish
        # Add websocket disconnection logic here if needed (if you have a way to close the connection from the client side)
        pass
# File: data/mexc_data_ingestion.py

import asyncio
import json
import logging
import websockets
import os
import hmac
import hashlib
import time
from dotenv import load_dotenv
from models.utils.config import Config
import websockets.exceptions as ws_exceptions

load_dotenv('.env')  # Load environment variables from .env file

class DataIngestion:
    def __init__(self, gmn: CryptoGMN, config: Config):
        self.config = config
        self.gmn = gmn
        self.symbol = config.symbol
        self.interval = config.interval
        self.ws_url = os.getenv('MEXC_WS_URL', 'wss://contract.mexc.com/ws')
        self.api_key = os.getenv('MEXC_API_KEY')
        self.api_secret = os.getenv('MEXC_API_SECRET')

    async def connect(self):
        while True:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    logging.info(f"Connected to {self.ws_url}")

                    if self.api_key and self.api_secret:
                        await self._login(websocket)

                    await self._subscribe_public_channels(websocket)
                    await self._receive_data_loop(websocket)

            except ws_exceptions.ConnectionClosedOK as e:
                logging.warning(f"WebSocket connection closed gracefully: {e.reason}")
                break
            except ws_exceptions.ConnectionClosedError as e:
                logging.error(f"WebSocket connection closed with error: {e.reason}")
                await asyncio.sleep(self.config.reconnect_delay)
                continue
            except Exception as e:
                logging.error(f"Connection error: {e}")
                await asyncio.sleep(self.config.reconnect_delay)
                continue

    async def _login(self, websocket):
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
        message = f"{self.api_key}{timestamp}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _subscribe_public_channels(self, websocket):
        subscription_message = {
            "method": "sub.kline",
            "param": {"symbol": self.symbol, "interval": self.interval},
            "id": 1
        }
        await websocket.send(json.dumps(subscription_message))
        await asyncio.sleep(1)  # Brief pause to allow subscription to process
        logging.info(f"Subscribed to {self.symbol} {self.interval} kline data.")

    async def _receive_data_loop(self, websocket):
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                await self._process_data(data)
            except ws_exceptions.ConnectionClosedOK as e:
                logging.warning(f"WebSocket connection closed gracefully: {e.reason}")
                break
            except ws_exceptions.ConnectionClosedError as e:
                logging.error(f"WebSocket connection closed with error: {e.reason}")
                await asyncio.sleep(self.config.reconnect_delay)
                break
            except json.JSONDecodeError:
                logging.error("Received invalid JSON data.")
            except Exception as e:
                logging.error(f"Error in receive_data_loop: {e}", exc_info=True)
                break

    async def _process_data(self, data: Dict):
        if 'data' in data and 'channel' in data and data['channel'] == 'push.kline':
            await self._process_kline_data(data['data'])

    async def _process_kline_data(self, kline_data):
        try:
            if isinstance(kline_data, list):
                # Assuming kline_data is a list of candle dictionaries
                await self.gmn.update_graph(kline_data)
                logging.debug(f"Updated GMN with kline data: {kline_data}")
            elif isinstance(kline_data, dict):
                # Single candle data
                await self.gmn.update_graph([kline_data])
                logging.debug(f"Updated GMN with kline data: {kline_data}")
            else:
                logging.warning("Received kline data in unexpected format.")
        except Exception as e:
            logging.error(f"Error updating GMN with kline data: {e}")

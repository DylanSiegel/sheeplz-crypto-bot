import asyncio
import json
import logging
import websockets
import pandas as pd
import os
import hmac
import hashlib
import time
from dotenv import load_dotenv
from models.gmn.gmn import GraphMetanetwork

# Load environment variables
load_dotenv('configs/secrets.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIngestion:
    def __init__(self, symbol=None, interval=None):
        self.symbol = symbol or os.getenv('DEFAULT_TRADING_SYMBOL', 'BTC_USDT')
        self.interval = interval or os.getenv('DEFAULT_KLINE_INTERVAL', 'Min1')
        self.ws_url = os.getenv('MEXC_WS_URL', 'wss://contract.mexc.com/edge')
        self.api_key = os.getenv('MEXC_API_KEY')
        self.api_secret = os.getenv('MEXC_API_SECRET')
        
        self.gmn = GraphMetanetwork()
        self.gmn.initialize_nodes(
            time_frames=["1m", "5m", "1h", "1d"],
            indicators=["price", "volume", "rsi", "macd", "fibonacci"]
        )

    async def connect(self):
        while True:  # Reconnection loop
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    logging.info(f"Connected to {self.ws_url}")

                    if self.api_key and self.api_secret:
                        await self.login(websocket)

                    # Subscribe to public channels
                    await self.subscribe_public_channels(websocket)

                    # Keep-alive loop
                    keep_alive_task = asyncio.create_task(self.keep_alive(websocket))

                    try:
                        while True:
                            data = await websocket.recv()
                            await self.process_data(json.loads(data))
                    except websockets.exceptions.ConnectionClosed:
                        logging.error("WebSocket connection closed")
                    finally:
                        keep_alive_task.cancel()
            except Exception as e:
                logging.error(f"Connection error: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting

    async def login(self, websocket):
        timestamp = int(time.time() * 1000)
        signature = self.generate_signature(timestamp)
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

    def generate_signature(self, timestamp):
        message = f"{self.api_key}{timestamp}"
        signature = hmac.new(self.api_secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()
        return signature

    async def subscribe_public_channels(self, websocket):
        subscriptions = [
            {"method": "kline.subscribe", "param": {"symbol": self.symbol, "interval": self.interval}},
            {"method": "deal.subscribe", "param": {"symbol": self.symbol}},
            {"method": "depth.subscribe", "param": {"symbol": self.symbol}},
            {"method": "index.subscribe", "param": {"symbol": self.symbol}},
            {"method": "fair.subscribe", "param": {"symbol": self.symbol}},
            {"method": "funding.subscribe", "param": {"symbol": self.symbol}}
        ]
        for sub in subscriptions:
            await websocket.send(json.dumps(sub))
            response = await websocket.recv()
            logging.info(f"Subscription response: {response}")

    async def keep_alive(self, websocket):
        while True:
            await websocket.send('ping')
            await asyncio.sleep(15)  # Send ping every 15 seconds

    async def process_data(self, data):
        try:
            if 'channel' in data:
                if data['channel'] == 'kline':
                    await self.process_kline_data(data['data'])
                elif data['channel'] == 'deal':
                    await self.process_deal_data(data['data'])
                elif data['channel'] == 'depth':
                    await self.process_depth_data(data['data'])
                elif data['channel'] in ['index', 'fair', 'funding']:
                    await self.process_market_data(data['channel'], data['data'])
            else:
                logging.warning(f"Received unexpected data format: {data}")
        except Exception as e:
            logging.error(f"Error processing data: {e}")

    async def process_kline_data(self, kline_data):
        df = pd.DataFrame(kline_data)
        os.makedirs(os.path.dirname(f"data/raw/kline_{self.symbol}_{self.interval}.csv"), exist_ok=True)
        df.to_csv(f"data/raw/kline_{self.symbol}_{self.interval}.csv", mode='a', header=False, index=False)
        logging.info(f"Kline data received and stored for {self.symbol}")
        
        market_data = {
            "price": df[['c']].astype(float),
            "volume": df[['v']].astype(float),
            "rsi": await self.calculate_rsi(df),
            "macd": await self.calculate_macd(df),
            "fibonacci": await self.calculate_fibonacci(df)
        }
        self.gmn.update_graph(market_data)
        logging.info(f"GMN updated with new kline data for {self.symbol}")

    async def process_deal_data(self, deal_data):
        df = pd.DataFrame(deal_data)
        os.makedirs(os.path.dirname(f"data/raw/deals_{self.symbol}.csv"), exist_ok=True)
        df.to_csv(f"data/raw/deals_{self.symbol}.csv", mode='a', header=False, index=False)
        logging.info(f"Deal data received and stored for {self.symbol}")

    async def process_depth_data(self, depth_data):
        # Implement order book maintenance logic here
        # For now, we'll just log the received data
        logging.info(f"Depth data received for {self.symbol}")

    async def process_market_data(self, channel, data):
        df = pd.DataFrame([data])
        os.makedirs(os.path.dirname(f"data/raw/{channel}_{self.symbol}.csv"), exist_ok=True)
        df.to_csv(f"data/raw/{channel}_{self.symbol}.csv", mode='a', header=False, index=False)
        logging.info(f"{channel.capitalize()} data received and stored for {self.symbol}")

    async def calculate_rsi(self, df):
        # Placeholder for RSI calculation
        # In a real implementation, you would calculate RSI based on the closing prices
        return pd.DataFrame({'rsi': [50] * len(df)})

    async def calculate_macd(self, df):
        # Placeholder for MACD calculation
        # In a real implementation, you would calculate MACD based on the closing prices
        return pd.DataFrame({'macd': [0] * len(df)})

    async def calculate_fibonacci(self, df):
        # Placeholder for Fibonacci levels calculation
        # In a real implementation, you would calculate Fibonacci levels based on price action
        return pd.DataFrame({'fibonacci': [0] * len(df)})

if __name__ == "__main__":
    # Example of how to use the class with different symbols/intervals
    data_ingestion_btc = DataIngestion(symbol="BTC_USDT", interval="Min5")
    data_ingestion_eth = DataIngestion(symbol="ETH_USDT", interval="Min15")

    # Run multiple instances concurrently
    asyncio.get_event_loop().run_until_complete(asyncio.gather(
        data_ingestion_btc.connect(),
        data_ingestion_eth.connect()
    ))
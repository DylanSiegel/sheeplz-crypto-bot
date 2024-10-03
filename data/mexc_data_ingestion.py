import asyncio
import websockets
import json
import pandas as pd
from configs.api_config import API_KEY, API_SECRET, BASE_URL
from models.gmn.gmn import GraphMetanetwork

class DataIngestion:
    def __init__(self, symbol="BTC_USDT", interval="1m"):
        self.symbol = symbol
        self.interval = interval
        self.ws_url = f"wss://wbs.mexc.com/ws"
        
        # Initialize the GMN
        self.gmn = GraphMetanetwork()
        self.gmn.initialize_nodes(
            time_frames=["1m", "5m", "1h", "1d"],
            indicators=["price", "volume", "rsi", "macd", "fibonacci"]
        )

    async def connect(self):
        async with websockets.connect(self.ws_url) as websocket:
            # Subscribe to candlestick data
            subscribe_message = json.dumps({
                "op": "sub.kline",
                "symbol": self.symbol,
                "interval": self.interval
            })
            await websocket.send(subscribe_message)

            while True:
                try:
                    data = await websocket.recv()
                    self.process_data(json.loads(data))
                except Exception as e:
                    print(f"Error receiving data: {e}")
                    break

    def process_data(self, data):
        # Process incoming data and update GMN
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            df.to_csv(f"data/raw/{self.symbol}_{self.interval}.csv", mode='a', header=False)
            print(f"Data received and stored for {self.symbol} at interval {self.interval}")
            
            # Calculate technical indicators here and store in market_data dict
            market_data = {
                "price": df[['price']],
                "volume": df[['volume']],
                # Calculate RSI, MACD, Fibonacci levels...
                "rsi": self.calculate_rsi(df),
                "macd": self.calculate_macd(df),
                "fibonacci": self.calculate_fibonacci(df)
            }
            self.gmn.update_graph(market_data)
            print(f"GMN updated with new data for {self.symbol} at interval {self.interval}")

    def calculate_rsi(self, df):
        # Implement RSI calculation
        # This is a placeholder implementation
        return pd.DataFrame({'rsi': [50] * len(df)})

    def calculate_macd(self, df):
        # Implement MACD calculation
        # This is a placeholder implementation
        return pd.DataFrame({'macd': [0] * len(df)})

    def calculate_fibonacci(self, df):
        # Implement Fibonacci levels calculation
        # This is a placeholder implementation
        return pd.DataFrame({'fibonacci': [0] * len(df)})

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    asyncio.run(data_ingestion.connect())
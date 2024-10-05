import asyncio
import logging
import os
from dotenv import load_dotenv
from data.mexc_data_ingestion import Config, DataIngestion
from models.gmn.gmn import CryptoGMN

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestGMN(CryptoGMN):
    def __init__(self, timeframes, indicators, max_history_length=1000):
        super().__init__(timeframes, indicators, max_history_length)
        self.received_data = []

    async def update_graph(self, new_data_items):
        await super().update_graph(new_data_items)
        self.received_data.extend(new_data_items)
        logging.info(f"Received {len(new_data_items)} new data items. Total: {len(self.received_data)}")

async def test_data_ingestion():
    # Configuration
    config = Config(
        symbol="BTC_USDT",
        timeframes=["1m", "5m", "15m"],
        private_channels=[],  # No private channels for this test
        reconnect_delay=5,
        max_reconnect_delay=60,
        backoff_factor=2.0,
        rate_limit=100,
        processing_queue_size=1000
    )

    # Initialize TestGMN
    gmn = TestGMN(config.timeframes, ["price", "volume", "rsi", "macd", "fibonacci"])

    # Initialize DataIngestion
    data_ingestion = DataIngestion(gmn=gmn, config=config)

    # Run the ingestion process for a fixed duration
    ingestion_task = asyncio.create_task(data_ingestion.connect())

    # Let it run for 5 minutes
    await asyncio.sleep(300)

    # Stop the ingestion process
    ingestion_task.cancel()
    await data_ingestion.close()

    # Analyze the received data
    logging.info(f"Total data points received: {len(gmn.received_data)}")

    # Check if data was received for all timeframes
    timeframe_counts = {tf: 0 for tf in config.timeframes}
    for item in gmn.received_data:
        if 'interval' in item:
            timeframe_counts[item['interval']] += 1

    for tf, count in timeframe_counts.items():
        logging.info(f"Data points for {tf}: {count}")
        assert count > 0, f"No data received for timeframe {tf}"

    # Check if all required fields are present in the data
    required_fields = ['t', 'o', 'h', 'l', 'c', 'v']
    for item in gmn.received_data[:10]:  # Check first 10 items
        for field in required_fields:
            assert field in item, f"Field {field} missing in data: {item}"

    # Check if RSI, MACD, and Fibonacci levels are calculated
    for tf in config.timeframes:
        assert len(gmn.get_data(f"{tf}_rsi")) > 0, f"No RSI data for {tf}"
        assert len(gmn.get_data(f"{tf}_macd")) > 0, f"No MACD data for {tf}"
        assert len(gmn.get_data(f"{tf}_fibonacci")) > 0, f"No Fibonacci data for {tf}"

    logging.info("All tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_data_ingestion())
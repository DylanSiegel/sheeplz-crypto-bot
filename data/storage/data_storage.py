import asyncio
import pandas as pd
import os

class DataStorage:
    def __init__(self, storage_path="data_storage"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    async def store_data(self, unified_feed):
        """Stores unified feed data asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._store_data_sync, unified_feed)

    def _store_data_sync(self, unified_feed):
        """Synchronous helper function for data storage."""
        symbol = "BTC_USDT"  # Assuming single symbol; adjust if needed.
        for timeframe, content in unified_feed.items():
            df = pd.DataFrame(content)  # Create DataFrame from content
            filename = f"{symbol}_{timeframe}.csv"
            filepath = os.path.join(self.storage_path, filename)
            df.to_csv(filepath, index=False)
            print(f"Data stored in {filepath}")

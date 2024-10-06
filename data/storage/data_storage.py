# File: data/storage/data_storage.py
import asyncio
import pandas as pd
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from error_handler import ErrorHandler
import logging

load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))

class DataStorage:
    """
    Handles storage of processed data. Supports multiple storage backends.
    """

    def __init__(self, storage_path: str = None, db_config: Dict[str, Any] = None, error_handler: ErrorHandler = None):
        """
        Initializes the DataStorage.
        
        Args:
            storage_path (str, optional): Path for file-based storage. Defaults to environment variable.
            db_config (Dict[str, Any], optional): Configuration for database storage. Defaults to None.
            error_handler (ErrorHandler, optional): Instance to handle errors. Defaults to None.
        """
        self.storage_path = storage_path or os.getenv("DATA_STORAGE_PATH", "./data_storage")
        os.makedirs(self.storage_path, exist_ok=True)
        self.db_config = db_config  # For future expansion to database storage
        self.error_handler = error_handler
        self.logger = logging.getLogger("DataStorage")
        self.file_locks = {}  # To prevent concurrent writes to the same file

    async def store_data(self, unified_feed: Dict[str, Any]):
        """
        Stores unified feed data asynchronously.
        
        Args:
            unified_feed (Dict[str, Any]): Processed data ready for storage.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._store_data_sync, unified_feed)

    def _store_data_sync(self, unified_feed: Dict[str, Any]):
        """
        Synchronously stores data to the chosen storage backend.
        
        Args:
            unified_feed (Dict[str, Any]): Processed data ready for storage.
        """
        symbol = unified_feed.get('symbol', 'UNKNOWN')
        timeframe = unified_feed.get('timeframe', 'UNKNOWN')
        data = unified_feed.get('data', [])
        indicators = unified_feed.get('indicators', {})

        if not data:
            self.logger.warning(f"No data to store for {symbol} {timeframe}")
            return

        # For file-based storage (CSV)
        self._store_to_csv(symbol, timeframe, data, indicators)

        # For database storage, implement respective methods here
        # Example: self._store_to_postgresql(symbol, timeframe, data, indicators)

    def _store_to_csv(self, symbol: str, timeframe: str, data: List[Dict[str, Any]], indicators: Dict[str, Any]):
        """
        Stores data to a CSV file.

        Args:
            symbol (str): Trading symbol.
            timeframe (str): Kline timeframe.
            data (List[Dict[str, Any]]): List of kline data records.
            indicators (Dict[str, Any]): Calculated technical indicators.
        """
        try:
            df = pd.DataFrame(data)
            # Add indicators to DataFrame
            for key, value in indicators.items():
                if isinstance(value, list) and len(value) == len(df):
                    df[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list) and len(sub_value) == len(df):
                            df[f"{key}_{sub_key}"] = sub_value
                else:
                    # For scalar or mismatched lengths, skip or handle accordingly
                    self.logger.debug(f"Skipping indicator {key} due to mismatched data length.")

            filename = f"{symbol}_{timeframe}.csv"
            filepath = os.path.join(self.storage_path, filename)

            # Acquire a lock for the file to prevent concurrent writes
            lock = self.file_locks.get(filepath)
            if not lock:
                lock = asyncio.Lock()
                self.file_locks[filepath] = lock

            async def write_csv():
                if os.path.exists(filepath):
                    existing_df = pd.read_csv(filepath)
                    df_combined = pd.concat([existing_df, df], ignore_index=True)
                    df_combined.drop_duplicates(subset=['close_time'], keep='last', inplace=True)
                    df_combined.sort_values(by='close_time', inplace=True)
                    df_combined.to_csv(filepath, index=False)
                else:
                    df.to_csv(filepath, index=False)
                self.logger.info(f"Data stored in {filepath}")

            # Schedule the write operation in the event loop
            asyncio.get_event_loop().create_task(self._write_with_lock(lock, write_csv))

        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(f"Error storing data to {filepath}: {e}", exc_info=True, symbol=symbol, timeframe=timeframe)
            else:
                self.logger.error(f"Error storing data to {filepath}: {e}")

    async def _write_with_lock(self, lock: asyncio.Lock, write_coroutine):
        """
        Ensures that writing to a file is thread-safe using asyncio locks.

        Args:
            lock (asyncio.Lock): The lock associated with the file.
            write_coroutine (callable): The coroutine to execute for writing.
        """
        async with lock:
            await write_coroutine()

    async def load_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Loads existing DataFrame from storage.

        Args:
            symbol (str): Trading symbol.
            timeframe (str): Kline timeframe.

        Returns:
            pd.DataFrame: Loaded DataFrame or empty DataFrame if file doesn't exist.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._load_dataframe_sync, symbol, timeframe)

    def _load_dataframe_sync(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Synchronously loads DataFrame from storage.

        Args:
            symbol (str): Trading symbol.
            timeframe (str): Kline timeframe.

        Returns:
            pd.DataFrame: Loaded DataFrame or empty DataFrame if file doesn't exist.
        """
        filename = f"{symbol}_{timeframe}.csv"
        filepath = os.path.join(self.storage_path, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                self.logger.debug(f"Loaded existing data from {filepath}")
                return df
            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle_error(f"Error loading data from {filepath}: {e}", exc_info=True, symbol=symbol, timeframe=timeframe)
                self.logger.error(f"Error loading data from {filepath}: {e}")
                return pd.DataFrame()
        else:
            self.logger.info(f"No existing data found for {symbol} {timeframe}. Starting fresh.")
            return pd.DataFrame()

    # Placeholder methods for database storage implementations
    def _store_to_postgresql(self, symbol: str, timeframe: str, data: List[Dict[str, Any]], indicators: Dict[str, Any]):
        """
        Stores data to a PostgreSQL database.

        Args:
            symbol (str): Trading symbol.
            timeframe (str): Kline timeframe.
            data (List[Dict[str, Any]]): List of kline data records.
            indicators (Dict[str, Any]): Calculated technical indicators.
        """
        # Implement PostgreSQL storage logic here
        pass

    def _store_to_mongodb(self, symbol: str, timeframe: str, data: List[Dict[str, Any]], indicators: Dict[str, Any]):
        """
        Stores data to a MongoDB database.

        Args:
            symbol (str): Trading symbol.
            timeframe (str): Kline timeframe.
            data (List[Dict[str, Any]]): List of kline data records.
            indicators (Dict[str, Any]): Calculated technical indicators.
        """
        # Implement MongoDB storage logic here
        pass

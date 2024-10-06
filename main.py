import asyncio
from data.mexc_websocket_connector import MexcWebsocketConnector
from data.data_processor import DataProcessor
from data.indicator_calculations import IndicatorCalculator
from data.storage.data_storage import DataStorage
from data.storage.data_storage_postgresql import DataStoragePostgreSQL  # If using PostgreSQL
from error_handler import ErrorHandler
from dotenv import load_dotenv
import os

def load_configuration():
    """
    Loads configuration from the .env file.

    Returns:
        Tuple[List[str], List[str], str]: Symbols, timeframes, storage path.
    """
    load_dotenv(os.path.join(os.path.dirname(__file__), 'configs/.env'))
    symbols = os.getenv("SYMBOLS", "BTCUSDT").split(",")
    timeframes = os.getenv("TIMEFRAMES", "1m,15m,30m,4h,1d,1w,1M").split(",")
    storage_path = os.getenv("DATA_STORAGE_PATH", "./data_storage")
    return symbols, timeframes, storage_path

async def main():
    """
    Main entry point for the data pipeline.
    """
    symbols, timeframes, storage_path = load_configuration()

    # Initialize components
    data_queue = asyncio.Queue()
    error_handler = ErrorHandler()
    
    # Choose storage method:
    # For CSV storage:
    storage = DataStorage(storage_path=storage_path)
    
    # For PostgreSQL storage:
    # storage = DataStoragePostgreSQL()
    
    indicator_calculator = IndicatorCalculator(error_handler=error_handler)
    processor = DataProcessor(data_queue, storage, indicator_calculator, error_handler, symbols, timeframes)
    connector = MexcWebsocketConnector(data_queue, symbols, timeframes, error_handler)

    # Create tasks
    tasks = [
        asyncio.create_task(connector.connect()),
        asyncio.create_task(processor.run())
    ]

    # Run tasks until complete
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Pipeline terminated by user.")

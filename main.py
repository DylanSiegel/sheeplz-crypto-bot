# File: main.py
import asyncio
import logging
from data.mexc_websocket_connector import MexcWebsocketConnector
from data.data_processor import DataProcessor
from data.indicator_calculations import IndicatorCalculator
from data.storage.data_storage import DataStorage
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
    timeframes = os.getenv("TIMEFRAMES", "Min1,Min5,Min15,Min30,Hour4,Hour8,Day1,Week1,Month1").split(",")
    storage_path = os.getenv("DATA_STORAGE_PATH", "./data_storage")
    return symbols, timeframes, storage_path

async def main():
    """
    Main entry point for the data pipeline.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Main")
    
    symbols, timeframes, storage_path = load_configuration()
    logger.info(f"Loaded configuration: Symbols={symbols}, Timeframes={timeframes}, Storage Path={storage_path}")

    # Initialize components
    data_queue = asyncio.Queue()
    error_handler = ErrorHandler()
    
    # Choose storage method:
    # For CSV storage:
    storage = DataStorage(storage_path=storage_path, error_handler=error_handler)
    
    # For PostgreSQL storage:
    # storage = DataStoragePostgreSQL()
    
    indicator_calculator = IndicatorCalculator(error_handler=error_handler)
    processor = DataProcessor(data_queue, storage, indicator_calculator, error_handler, symbols, timeframes)
    connector = MexcWebsocketConnector(data_queue, symbols, timeframes, error_handler)

    # Create tasks
    connector_task = asyncio.create_task(connector.connect())
    processor_task = asyncio.create_task(processor.run())

    try:
        # Await both tasks concurrently
        await asyncio.gather(connector_task, processor_task)
    except asyncio.CancelledError:
        logger.info("Tasks cancelled. Shutting down...")
    except Exception as e:
        error_handler.handle_error(f"Unexpected error in main: {e}", exc_info=True)
    finally:
        # Cancel tasks if they are still running
        connector_task.cancel()
        processor_task.cancel()
        await asyncio.gather(connector_task, processor_task, return_exceptions=True)
        logger.info("Data pipeline terminated gracefully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger("Main").info("Pipeline terminated by user.")

# File: main.py
import asyncio
import logging
import os
from dotenv import load_dotenv
from data.mexc_websocket_connector import MexcWebsocketConnector
from data.data_processor import DataProcessor
from error_handler import ErrorHandler

def load_configuration():
    load_dotenv(os.path.join(os.path.dirname(__file__), 'configs/.env'))
    symbols = os.getenv("SYMBOLS", "BTCUSDT").split(",")
    timeframes = os.getenv("TIMEFRAMES", "Min1").split(",")
    return symbols, timeframes

def validate_configuration(symbols, timeframes):
    if not symbols or symbols == ['']:
        raise ValueError("SYMBOLS must be defined in the .env file.")
    if not timeframes or timeframes == ['']:
        raise ValueError("TIMEFRAMES must be defined in the .env file.")

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Change to logging.DEBUG for more verbosity
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("Main")

    symbols, timeframes = load_configuration()
    try:
        validate_configuration(symbols, timeframes)
    except ValueError as ve:
        logger.error(f"Configuration Error: {ve}")
        return
    logger.info(f"Loaded configuration: Symbols={symbols}, Timeframes={timeframes}")

    # Initialize components
    data_queue = asyncio.Queue()
    error_handler = ErrorHandler()

    processor = DataProcessor(data_queue, error_handler, symbols, timeframes)
    connector = MexcWebsocketConnector(data_queue, symbols, timeframes, error_handler)

    # Create tasks
    connector_task = asyncio.create_task(connector.connect())
    processor_task = asyncio.create_task(processor.run_lnn())  # Run the LNN processing loop

    tasks = [connector_task, processor_task]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Pipeline terminated by user.")
    except Exception as e:
        error_handler.handle_error(f"Unexpected error in main: {e}", exc_info=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())

import pytest
import asyncio
from data.data_processor import DataProcessor
from data.config import Config
from data.indicator_calculations import IndicatorCalculator
from data.error_handler import ErrorHandler
from data.storage.data_storage import DataStorage
from data.mexc_websocket_connector import MexcWebsocketConnector


@pytest.mark.asyncio
async def test_realtime_mexc_data_pipeline():
    """
    Test the data pipeline with real-time data from the MEXC WebSocket API.
    It connects to the WebSocket, processes incoming kline data in real-time,
    and validates that the unified feed is being generated correctly.
    """
    # Step 1: Set up Config, ErrorHandler, and IndicatorCalculator
    config = Config()
    error_handler = ErrorHandler()
    indicator_calculator = IndicatorCalculator(error_handler)

    # Step 2: Initialize DataStorage to store processed data
    data_storage = DataStorage()

    # Step 3: Initialize DataProcessor with real components
    processor = DataProcessor(data_storage, indicator_calculator, error_handler, config)

    # Step 4: Set up WebSocket connection to MEXC
    ws_data_queue = asyncio.Queue()  # Queue for processing incoming data
    mexc_connector = MexcWebsocketConnector(config, ws_data_queue)

    # Step 5: Function to process real-time data
    async def process_real_time_data():
        while True:
            try:
                data_batch = await ws_data_queue.get()
                if data_batch:  # Ensure there's data to process
                    await processor.process_data(data_batch)
            except Exception as e:
                print(f"Error while processing real-time data: {e}")
                break

    # Step 6: Start WebSocket connection and data processing concurrently
    websocket_task = asyncio.create_task(mexc_connector.connect())
    data_processing_task = asyncio.create_task(process_real_time_data())

    # Step 7: Allow the tasks to run for a certain amount of time (e.g., 2 minutes) for test purposes
    await asyncio.sleep(120)  # Adjust sleep time for a longer test duration if needed

    # Step 8: Cancel the tasks after the test period
    websocket_task.cancel()
    data_processing_task.cancel()

    # Step 9: Validate stored data after 2 minutes of real-time data collection
    symbol = "BTC_USDT"
    unified_feed_1m = data_storage.get_data(symbol, "1m")
    unified_feed_5m = data_storage.get_data(symbol, "5m")

    # Ensure the unified feed contains data after processing real-time WebSocket data
    assert not unified_feed_1m.empty, "Unified feed for 1m timeframe is empty"
    assert not unified_feed_5m.empty, "Unified feed for 5m timeframe is empty"

    # Print some of the collected data for manual inspection
    print(f"Unified feed for 1m: {unified_feed_1m.head()}")
    print(f"Unified feed for 5m: {unified_feed_5m.head()}")

    # Validate specific indicators in the unified feed
    validate_indicators(unified_feed_1m)
    validate_indicators(unified_feed_5m)


def validate_indicators(unified_feed):
    """Helper function to validate the indicator contents."""
    assert 'rsi' in unified_feed.columns, "Missing RSI indicator"
    assert 'macd' in unified_feed.columns, "Missing MACD indicator"
    assert 'fibonacci' in unified_feed.columns, "Missing Fibonacci indicator"

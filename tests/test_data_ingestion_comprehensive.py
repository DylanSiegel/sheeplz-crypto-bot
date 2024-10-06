import asyncio
import pytest
import logging
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

from data.config import Config as DataIngestionConfig
from data.mexc_websocket_connector import MexcWebsocketConnector
from data.data_processor import DataProcessor
from models.gmn.crypto_gmn import CryptoGMN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def event_loop():
    """Create a new event loop for the tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def config():
    """Load data ingestion configuration from .env file."""
    return DataIngestionConfig()

@pytest.fixture(scope="module")
def gmn(config):
    """Create a CryptoGMN instance using the loaded configuration."""
    return CryptoGMN(config)

@pytest.fixture(scope="module")
async def data_queue():
    """Create an asynchronous queue to store data received from the websocket."""
    return asyncio.Queue()

@pytest.fixture(scope="module")
async def websocket_connector(config, data_queue):
    """Create a MexcWebsocketConnector instance and connect to the websocket."""
    connector = MexcWebsocketConnector(config, data_queue)
    await connector.connect()
    yield connector
    await connector.close()

@pytest.fixture(scope="module")
def data_processor(gmn):
    """Create a DataProcessor instance using the CryptoGMN instance."""
    return DataProcessor(gmn)

@pytest.mark.asyncio
async def test_websocket_connection(websocket_connector):
    """Verify that the websocket connection is established successfully."""
    assert websocket_connector.connected_event.is_set()

@pytest.mark.asyncio
async def test_websocket_reconnection(config, data_queue, monkeypatch):
    """Test the reconnection logic of the websocket connector."""
    mock_websocket = AsyncMock()
    mock_websocket.connect.side_effect = [ConnectionError, None]
    monkeypatch.setattr("websockets.connect", mock_websocket)

    connector = MexcWebsocketConnector(config, data_queue)
    await connector.connect()

    assert connector.connected_event.is_set()
    assert mock_websocket.connect.call_count == 2

@pytest.mark.asyncio
async def test_data_reception(websocket_connector, data_queue):
    """Verify that data is received correctly from the websocket."""
    test_data = {"channel": "push.kline", "data": {"c": 50000, "v": 1.5}}
    await websocket_connector.on_message(test_data)

    received_data = await asyncio.wait_for(data_queue.get(), timeout=1.0)
    assert received_data == test_data

@pytest.mark.asyncio
@pytest.mark.parametrize("test_data", [
    {"channel": "push.kline", "data": {"c": 50000, "v": 1.5}},
    {"channel": "push.ticker", "data": {"last": 51000, "vol": 100}},
])
async def test_data_processing(data_processor, test_data):
    """Verify that the DataProcessor correctly processes different types of data."""
    await data_processor.process_data([test_data])

    if test_data["channel"] == "push.kline":
        assert data_processor.gmn.get_data("1m", "price")[-1] == test_data["data"]["c"]
        assert data_processor.gmn.get_data("1m", "volume")[-1] == test_data["data"]["v"]
    elif test_data["channel"] == "push.ticker":
        assert data_processor.gmn.get_data("1m", "price")[-1] == test_data["data"]["last"]
        assert data_processor.gmn.get_data("1m", "volume")[-1] == test_data["data"]["vol"]

@pytest.mark.asyncio
async def test_gmn_updates(gmn, data_processor):
    """Verify that the CryptoGMN instance is updated correctly with new data."""
    test_data = {"channel": "push.kline", "data": {"c": 52000, "v": 2.0}}
    await data_processor.process_data([test_data])

    assert gmn.get_data("1m", "price")[-1] == 52000
    assert gmn.get_data("1m", "volume")[-1] == 2.0

@pytest.mark.asyncio
async def test_indicator_calculation(gmn, data_processor):
    """Test that indicators are calculated correctly when enough data is available."""
    # Simulate receiving enough data points for indicator calculation
    for i in range(20):
        test_data = {"channel": "push.kline", "data": {"c": 50000 + i * 100, "v": 1.5 + i * 0.1}}
        await data_processor.process_data([test_data])

    assert len(gmn.get_data("1m", "rsi")) > 0
    assert len(gmn.get_data("1m", "macd")) > 0
    assert len(gmn.get_data("1m", "fibonacci")) > 0

@pytest.mark.asyncio
async def test_error_handling(data_processor):
    """Test that the data processor handles errors gracefully."""
    invalid_data = {"channel": "push.kline", "data": "invalid"}

    with pytest.raises(Exception):
        await data_processor.process_data([invalid_data])

    # Verify that valid data can still be processed after an error
    valid_data = {"channel": "push.kline", "data": {"c": 53000, "v": 2.5}}
    await data_processor.process_data([valid_data])
    assert data_processor.gmn.get_data("1m", "price")[-1] == 53000

@pytest.mark.asyncio
async def test_multiple_timeframes(gmn, data_processor):
    """Test that data is correctly processed for multiple timeframes."""
    test_data = {"channel": "push.kline", "data": {"c": 54000, "v": 3.0}}
    await data_processor.process_data([test_data])

    for timeframe in gmn.timeframes:
        assert gmn.get_data(timeframe, "price")[-1] == 54000
        assert gmn.get_data(timeframe, "volume")[-1] == 3.0

@pytest.mark.asyncio
async def test_data_consistency(gmn, data_processor):
    """Test that data remains consistent across multiple updates."""
    initial_data = {"channel": "push.kline", "data": {"c": 55000, "v": 3.5}}
    await data_processor.process_data([initial_data])

    initial_price = gmn.get_data("1m", "price")[-1]
    initial_volume = gmn.get_data("1m", "volume")[-1]

    # Process more data
    for i in range(5):
        test_data = {"channel": "push.kline", "data": {"c": 55000 + i * 100, "v": 3.5 + i * 0.1}}
        await data_processor.process_data([test_data])

    # Check that initial data is still correct
    assert gmn.get_data("1m", "price")[0] == initial_price
    assert gmn.get_data("1m", "volume")[0] == initial_volume

@pytest.mark.asyncio
async def test_data_queue_overflow(config, data_queue):
    """Test that the data queue doesn't overflow."""
    connector = MexcWebsocketConnector(config, data_queue)
    
    # Fill the queue to its limit
    for i in range(config.processing_queue_size + 10):
        await connector.on_message({"data": f"test{i}"})

    assert data_queue.qsize() == config.processing_queue_size, "Queue should not exceed its maximum size"

@pytest.mark.asyncio
async def test_websocket_connection_established(websocket_connector):
    """Verify that the websocket connection is established successfully."""
    assert websocket_connector.connected_event.is_set(), "WebSocket should be connected"

@pytest.mark.asyncio
async def test_websocket_connection_closed(websocket_connector):
    """Verify that the websocket connection is closed successfully."""
    await websocket_connector.close()
    assert not websocket_connector.connected_event.is_set(), "WebSocket should be closed"

@pytest.mark.asyncio
async def test_data_processor_shutdown(data_processor):
    """Test that the data processor shuts down correctly."""
    await data_processor.shutdown()
    assert data_processor.shutdown_event.is_set(), "Data processor should be shut down"

@pytest.mark.asyncio
async def test_gmn_shutdown(gmn):
    """Test that the CryptoGMN instance shuts down correctly."""
    await gmn.shutdown()
    assert gmn.shutdown_event.is_set(), "CryptoGMN should be shut down"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
import websockets
import pandas as pd
import logging
from data.mexc_data_ingestion import DataIngestion  # Assume this is the correct import path

@pytest.fixture
def data_ingestion():
    return DataIngestion(symbol="BTC_USDT", interval="Min1")

@pytest.mark.asyncio
async def test_connect_and_process_data(data_ingestion):
    mock_websocket = AsyncMock()
    mock_websocket.recv.side_effect = [
        json.dumps({
            "data": [{
                "t": 1625097600000,
                "o": "35000.00",
                "h": "35100.00",
                "l": "34900.00",
                "c": "35050.00",
                "v": "100.5",
                "q": "3520250.00"
            }]
        }),
        websockets.exceptions.ConnectionClosed(1000, "Normal closure")
    ]

    with patch('websockets.connect', return_value=mock_websocket):
        with patch.object(data_ingestion, 'process_data', AsyncMock()) as mock_process_data:
            await data_ingestion.connect()

    assert mock_websocket.send.called
    mock_process_data.assert_called_once()

@pytest.mark.asyncio
async def test_websocket_reconnection(data_ingestion):
    mock_websocket = AsyncMock()
    mock_websocket.recv.side_effect = [
        websockets.exceptions.ConnectionClosed(1000, "Normal closure"),
        Exception("Stop test")
    ]

    with patch('websockets.connect', return_value=mock_websocket):
        with pytest.raises(Exception, match="Stop test"):
            await data_ingestion.connect()

    assert mock_websocket.send.call_count == 2  # Initial connection + 1 reconnection attempt

@pytest.mark.asyncio
async def test_invalid_data_handling(data_ingestion, caplog):
    invalid_data = {"invalid": "data"}
    
    with patch.object(data_ingestion.gmn, 'update_graph') as mock_update_graph:
        await data_ingestion.process_data(invalid_data)
        
    mock_update_graph.assert_not_called()
    assert "Received unexpected data format" in caplog.text

@pytest.mark.asyncio
async def test_valid_data_processing(data_ingestion):
    valid_data = {
        "data": [{
            "t": 1625097600000,
            "o": "35000.00",
            "h": "35100.00",
            "l": "34900.00",
            "c": "35050.00",
            "v": "100.5",
            "q": "3520250.00"
        }]
    }

    with patch.object(data_ingestion.gmn, 'update_graph') as mock_update_graph:
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            await data_ingestion.process_data(valid_data)

    mock_update_graph.assert_called_once()
    mock_to_csv.assert_called_once()

@pytest.mark.asyncio
async def test_technical_indicator_calculation(data_ingestion):
    valid_data = {
        "data": [{
            "t": 1625097600000,
            "o": "35000.00",
            "h": "35100.00",
            "l": "34900.00",
            "c": "35050.00",
            "v": "100.5",
            "q": "3520250.00"
        }]
    }

    with patch.object(data_ingestion.gmn, 'update_graph') as mock_update_graph:
        with patch.object(data_ingestion, 'calculate_rsi', return_value=pd.DataFrame({'rsi': [50]})):
            with patch.object(data_ingestion, 'calculate_macd', return_value=pd.DataFrame({'macd': [0]})):
                with patch.object(data_ingestion, 'calculate_fibonacci', return_value=pd.DataFrame({'fibonacci': [0]})):
                    await data_ingestion.process_data(valid_data)

    update_args = mock_update_graph.call_args[0][0]
    assert all(key in update_args for key in ['price', 'volume', 'rsi', 'macd', 'fibonacci'])

@pytest.mark.asyncio
async def test_error_handling_in_process_data(data_ingestion, caplog):
    invalid_data = {
        "data": [{
            "t": 1625097600000,
            # Missing some required fields
        }]
    }

    with patch.object(data_ingestion.gmn, 'update_graph') as mock_update_graph:
        await data_ingestion.process_data(invalid_data)

    mock_update_graph.assert_not_called()
    assert "Error processing data" in caplog.text

@pytest.mark.asyncio
async def test_calculate_rsi(data_ingestion):
    df = pd.DataFrame({'price': [100, 102, 98, 101, 99]})
    result = await data_ingestion.calculate_rsi(df)
    assert isinstance(result, pd.DataFrame)
    assert 'rsi' in result.columns

@pytest.mark.asyncio
async def test_calculate_macd(data_ingestion):
    df = pd.DataFrame({'price': [100, 102, 98, 101, 99]})
    result = await data_ingestion.calculate_macd(df)
    assert isinstance(result, pd.DataFrame)
    assert 'macd' in result.columns

@pytest.mark.asyncio
async def test_calculate_fibonacci(data_ingestion):
    df = pd.DataFrame({'price': [100, 102, 98, 101, 99]})
    result = await data_ingestion.calculate_fibonacci(df)
    assert isinstance(result, pd.DataFrame)
    assert 'fibonacci' in result.columns

if __name__ == "__main__":
    pytest.main()
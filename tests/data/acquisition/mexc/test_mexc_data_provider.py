# tests/data/acquisition/mexc/test_mexc_data_provider.py

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.data.acquisition.mexc.mexc_data_provider import MexcDataProvider
from src.data.acquisition.mexc.utils.mexc_config_reader import load_mexc_api_keys

# To run this test file individually, use:
# pytest tests/data/acquisition/mexc/test_mexc_data_provider.py -v

@pytest.fixture
async def mexc_provider_mock():
    provider = MexcDataProvider("fake_api_key", "fake_api_secret")
    yield provider
    await provider.close()

@pytest.fixture
async def mexc_provider_real():
    api_key, api_secret = load_mexc_api_keys()
    provider = MexcDataProvider(api_key, api_secret)
    yield provider
    await provider.close()

@pytest.mark.asyncio
async def test_get_historical_data_mock(mexc_provider_mock):
    with patch.object(mexc_provider_mock.rest_api, 'get_klines', new_callable=AsyncMock) as mock_get_klines:
        mock_get_klines.return_value = [{"open_time": 1625097600000, "open": "33000.0", "high": "33100.0", "low": "32900.0", "close": "33050.0", "volume": "100.5"}]
        
        result = await mexc_provider_mock.get_historical_data("BTCUSDT", "1h", 1625097600000, 1625184000000)
        
        assert len(result) == 1
        assert result[0]["open"] == "33000.0"
        mock_get_klines.assert_called_once_with("BTCUSDT", "1h", 1625097600000, 1625184000000)

@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_historical_data_real(mexc_provider_real):
    result = await mexc_provider_real.get_historical_data("BTCUSDT", "1h", 1625097600000, 1625184000000)
    
    assert len(result) > 0
    assert "open" in result[0]
    assert "high" in result[0]
    assert "low" in result[0]
    assert "close" in result[0]

@pytest.mark.asyncio
async def test_stream_data(mexc_provider_mock):
    mock_callback = Mock()
    
    with patch.object(mexc_provider_mock.websocket, 'subscribe_klines', new_callable=AsyncMock) as mock_subscribe_klines:
        await mexc_provider_mock.stream_data(["BTCUSDT"], mock_callback)
        
        mock_subscribe_klines.assert_called_once_with(["BTCUSDT"], mock_callback)

@pytest.mark.asyncio
async def test_get_account_info(mexc_provider_mock):
    with patch.object(mexc_provider_mock.rest_api, 'get_account_info', new_callable=AsyncMock) as mock_get_account_info:
        mock_get_account_info.return_value = {"balances": [{"asset": "BTC", "free": "1.0", "locked": "0.5"}]}
        
        result = await mexc_provider_mock.get_account_info()
        
        assert "balances" in result
        assert result["balances"][0]["asset"] == "BTC"
        mock_get_account_info.assert_called_once()

@pytest.mark.asyncio
async def test_place_order(mexc_provider_mock):
    with patch.object(mexc_provider_mock.rest_api, 'place_order', new_callable=AsyncMock) as mock_place_order:
        mock_place_order.return_value = {"orderId": "12345", "status": "FILLED"}
        
        result = await mexc_provider_mock.place_order("BTCUSDT", "BUY", "LIMIT", 0.1, 30000)
        
        assert result["orderId"] == "12345"
        assert result["status"] == "FILLED"
        mock_place_order.assert_called_once_with("BTCUSDT", "BUY", "LIMIT", 0.1, 30000)

@pytest.mark.asyncio
async def test_get_order_book(mexc_provider_mock):
    with patch.object(mexc_provider_mock.order_book, 'get_order_book', new_callable=AsyncMock) as mock_get_order_book:
        mock_get_order_book.return_value = {"bids": [["30000", "0.1"]], "asks": [["30100", "0.2"]]}
        
        result = await mexc_provider_mock.get_order_book("BTCUSDT", 5)
        
        assert "bids" in result
        assert "asks" in result
        mock_get_order_book.assert_called_once_with("BTCUSDT", 5)

@pytest.mark.asyncio
async def test_stream_order_book(mexc_provider_mock):
    mock_callback = Mock()
    
    with patch.object(mexc_provider_mock.websocket, 'subscribe_order_book', new_callable=AsyncMock) as mock_subscribe_order_book:
        await mexc_provider_mock.stream_order_book("BTCUSDT", mock_callback)
        
        mock_subscribe_order_book.assert_called_once_with("BTCUSDT", mock_callback)

@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_account_info_real(mexc_provider_real):
    result = await mexc_provider_real.get_account_info()
    
    assert "balances" in result
    assert isinstance(result["balances"], list)
    assert len(result["balances"]) > 0

# To run only unit tests:
# pytest tests/data/acquisition/mexc/test_mexc_data_provider.py -v -m "not integration"

# To run only integration tests:
# pytest tests/data/acquisition/mexc/test_mexc_data_provider.py -v -m "integration"

# To run all tests:
# pytest tests/data/acquisition/mexc/test_mexc_data_provider.py -v
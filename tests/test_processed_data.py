import pytest
from unittest.mock import AsyncMock, MagicMock
from data.data_processor import DataProcessor
from data.config import Config
from data.indicator_calculations import IndicatorCalculator
from data.error_handler import ErrorHandler


@pytest.mark.asyncio
async def test_processed_data():
    """
    Tests that DataProcessor correctly processes kline data, applies indicators,
    and creates a unified feed.
    """
    # Setup mock gmn with store_data as an AsyncMock
    mock_gmn = MagicMock()
    mock_gmn.store_data = AsyncMock()

    # Setup IndicatorCalculator with mock error_handler
    mock_error_handler = MagicMock(spec=ErrorHandler)
    indicator_calculator = IndicatorCalculator(mock_error_handler)

    # Initialize Config
    config = Config()

    # Initialize DataProcessor
    processor = DataProcessor(mock_gmn, indicator_calculator, mock_error_handler, config)

    # Sample kline data for multiple timeframes
    sample_kline_data = get_sample_kline_data()

    # Process the kline data
    await processor.process_data(sample_kline_data)

    # Ensure store_data was called
    assert mock_gmn.store_data.called, "store_data was not called"

    # Get the unified_feed argument
    unified_feed = mock_gmn.store_data.call_args[0][0]

    # Validate the unified feed
    validate_unified_feed(unified_feed)


def get_sample_kline_data():
    """Generates sample kline data for 1m and 5m timeframes."""
    return [
        {
            'method': 'some_method',
            'c': 'spot@public.kline.v3.api@1m',
            'd': {
                'k': {
                    'T': 1638316800000,
                    'a': 100.0,
                    'c': 30000.0,
                    'h': 30050.0,
                    'i': '1m',
                    'l': 29950.0,
                    'o': 29975.0,
                    't': 1638316740000,
                    'v': 50.0
                }
            }
        },
        {
            'method': 'some_method',
            'c': 'spot@public.kline.v3.api@5m',
            'd': {
                'k': {
                    'T': 1638317100000,
                    'a': 150.0,
                    'c': 30050.0,
                    'h': 30100.0,
                    'i': '5m',
                    'l': 30000.0,
                    'o': 30025.0,
                    't': 1638317040000,
                    'v': 75.0
                }
            }
        },
        {
            'method': 'PONG',  # Heartbeat message
            'c': 'some_other_channel',
            'd': {}
        }
    ]


def validate_unified_feed(unified_feed):
    """Helper function to validate the structure and content of the unified feed."""
    assert '1m' in unified_feed
    assert '5m' in unified_feed

    for timeframe in ['1m', '5m']:
        assert 'price' in unified_feed[timeframe]
        assert 'volume' in unified_feed[timeframe]
        assert 'open' in unified_feed[timeframe]
        assert 'high' in unified_feed[timeframe]
        assert 'low' in unified_feed[timeframe]
        assert 'close_time' in unified_feed[timeframe]
        assert 'open_time' in unified_feed[timeframe]
        assert 'quantity' in unified_feed[timeframe]
        assert 'indicators' in unified_feed[timeframe]

        # Validate indicators
        assert 'rsi' in unified_feed[timeframe]['indicators'], "Missing RSI indicator"
        assert 'macd' in unified_feed[timeframe]['indicators'], "Missing MACD indicator"
        assert 'fibonacci' in unified_feed[timeframe]['indicators'], "Missing Fibonacci indicator"

        # Validate the contents of '1m'
        if timeframe == '1m':
            assert unified_feed['1m']['price'] == [30000.0]
            assert unified_feed['1m']['volume'] == [50.0]
            assert unified_feed['1m']['open'] == [29975.0]
            assert unified_feed['1m']['high'] == [30050.0]
            assert unified_feed['1m']['low'] == [29950.0]
            assert unified_feed['1m']['quantity'] == [100.0]
            validate_indicators(unified_feed['1m']['indicators'])

        # Validate the contents of '5m'
        if timeframe == '5m':
            assert unified_feed['5m']['price'] == [30050.0]
            assert unified_feed['5m']['volume'] == [75.0]
            assert unified_feed['5m']['open'] == [30025.0]
            assert unified_feed['5m']['high'] == [30100.0]
            assert unified_feed['5m']['low'] == [30000.0]
            assert unified_feed['5m']['quantity'] == [150.0]
            validate_indicators(unified_feed['5m']['indicators'])


def validate_indicators(indicators):
    """Helper function to validate the indicator contents."""
    assert isinstance(indicators['rsi'], list), "RSI should be a list"
    assert isinstance(indicators['macd'], dict), "MACD should be a dictionary"
    assert 'macd' in indicators['macd'], "Missing MACD line"
    assert 'macd_signal' in indicators['macd'], "Missing MACD signal line"
    assert 'macd_hist' in indicators['macd'], "Missing MACD histogram"
    assert isinstance(indicators['fibonacci'], list), "Fibonacci should be a list"

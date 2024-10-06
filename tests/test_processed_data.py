# File: tests/test_processed_data.py

import asyncio
import pytest
from data.data_processor import DataProcessor
from models.gmn.crypto_gmn import CryptoGMN
from data.config import Config as DataIngestionConfig

@pytest.fixture(scope="module")
def config():
    """Load data ingestion configuration."""
    return DataIngestionConfig()

@pytest.fixture(scope="module")
def gmn(config):
    """Create a CryptoGMN instance."""
    return CryptoGMN(config.timeframes, config.max_history_length, 5)

@pytest.fixture(scope="module")
def data_processor(gmn):
    """Create a DataProcessor instance."""
    return DataProcessor(gmn)

@pytest.mark.asyncio
async def test_processed_data_sample(data_processor):
    """
    Test to check the processed data sample and ensure it's ready for GMN calculations.
    """

    # Example raw data batch that would be processed in the data flow
    raw_data_batch = [
        {'channel': 'push.kline', 'data': {'price': 50000, 'volume': 1.5}},
        {'channel': 'push.kline', 'data': {'price': 50100, 'volume': 1.6}},
        {'channel': 'push.kline', 'data': {'price': 50200, 'volume': 1.7}},
        {'channel': 'push.kline', 'data': {'price': 50300, 'volume': 1.8}},
        {'channel': 'push.kline', 'data': {'price': 50400, 'volume': 2.0}},
        # Add more data for multiple timeframes/indicators as necessary
    ]

    # Process the raw data through the data processor
    await data_processor.process_data(raw_data_batch)

    # Fetch the processed data from the GMN
    processed_data = data_processor.gmn.get_all_data()

    # Check if the processed data contains required indicators and is in expected format
    assert '1m' in processed_data, "Processed data missing '1m' timeframe"
    assert 'price' in processed_data['1m'], "Processed data missing 'price' indicator in '1m' timeframe"
    assert 'volume' in processed_data['1m'], "Processed data missing 'volume' indicator in '1m' timeframe"
    
    # Print the processed data for debugging and analysis
    print("Processed Data Sample for GMN Calculations:")
    print(processed_data)

    # Validate structure of processed data for GMN calculations
    assert isinstance(processed_data['1m']['price'], list), "Price data is not in the correct format (list)"
    assert isinstance(processed_data['1m']['volume'], list), "Volume data is not in the correct format (list)"
    assert len(processed_data['1m']['price']) > 0, "Price data list is empty"
    assert len(processed_data['1m']['volume']) > 0, "Volume data list is empty"

    # You can add further checks to ensure that all other indicators are processed properly

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_processed_data.py"])

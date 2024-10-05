# File: tests/test_crypto_gmn.py

import pytest
from models.gmn.gmn import CryptoGMN

@pytest.fixture
def gmn():
    timeframes = ["1m"]
    indicators = ["price", "volume", "rsi", "macd", "fibonacci"]
    return CryptoGMN(timeframes, indicators, max_history_length=100)

@pytest.mark.asyncio
async def test_gmn_data_update(gmn):
    new_data = [{
        "t": 1625097600000,
        "o": "35000.00",
        "h": "35100.00",
        "l": "34900.00",
        "c": "35050.00",
        "v": "100.5",
        "q": "3520250.00"
    }]
    await gmn.update_graph(new_data)

    price_data = gmn.get_data("1m", "price")
    volume_data = gmn.get_data("1m", "volume")

    assert price_data == [35050.00]
    assert volume_data == [100.5]

@pytest.mark.asyncio
async def test_gmn_indicator_calculations(gmn):
    # Insert enough data to calculate indicators
    for i in range(15):
        await gmn.update_graph([{
            "t": 1625097600000 + i * 60000,
            "o": "35000.00",
            "h": "35100.00",
            "l": "34900.00",
            "c": str(35050 + i),
            "v": "100.5",
            "q": "3520250.00"
        }])

    rsi_data = gmn.get_data("1m", "rsi")
    macd_data = gmn.get_data("1m", "macd")
    fibonacci_data = gmn.get_data("1m", "fibonacci")

    assert len(rsi_data) == 1
    assert len(macd_data) == 1
    assert len(fibonacci_data) == 1

@pytest.mark.asyncio
async def test_gmn_concurrent_updates(gmn):
    # Simulate concurrent updates
    new_data_1 = [{
        "t": 1625097600000,
        "o": "35000.00",
        "h": "35100.00",
        "l": "34900.00",
        "c": "35050.00",
        "v": "100.5",
        "q": "3520250.00"
    }]
    new_data_2 = [{
        "t": 1625097660000,
        "o": "35050.00",
        "h": "35150.00",
        "l": "34950.00",
        "c": "35100.00",
        "v": "101.0",
        "q": "3530250.00"
    }]

    await asyncio.gather(
        gmn.update_graph(new_data_1),
        gmn.update_graph(new_data_2)
    )

    price_data = gmn.get_data("1m", "price")
    volume_data = gmn.get_data("1m", "volume")

    assert price_data == [35050.00, 35100.00]
    assert volume_data == [100.5, 101.0]

# File: tests/test_rl_agent.py

import pytest
from models.agents.rl_agent import TradingEnvironment, train_rl_agent
import pandas as pd
from unittest.mock import patch

@pytest.fixture
def mock_market_data():
    data = {
        "price": [100, 102, 98, 101, 99, 105, 103],
        "volume": [10, 15, 10, 20, 15, 25, 20],
        "rsi": [50, 55, 45, 60, 50, 65, 55],
        "macd": [0.1, 0.2, -0.1, 0.3, 0.0, 0.4, 0.2],
        "fibonacci": [99, 100, 98, 101, 99, 105, 103]
    }
    return pd.DataFrame(data)

def test_trading_environment_reset(mock_market_data):
    env = TradingEnvironment(mock_market_data)
    obs = env.reset()
    assert env.current_step == 0
    assert len(obs) == len(mock_market_data.columns)

def test_trading_environment_step(mock_market_data):
    env = TradingEnvironment(mock_market_data)
    env.reset()
    obs, reward, done, info = env.step(0)  # Action: Buy
    assert env.current_step == 1
    assert not done
    assert len(obs) == len(mock_market_data.columns)

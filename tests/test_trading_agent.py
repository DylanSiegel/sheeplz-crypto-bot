# File: tests/test_trading_agent.py

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import torch
import numpy as np
from models.agents.agent import TradingAgent
from models.utils.config import Config
from models.utils.risk_management import RiskManager
from sklearn.preprocessing import MinMaxScaler

@pytest.fixture
def mock_config():
    config_data = {
        "threshold_buy": 0.7,
        "threshold_sell": 0.3,
        "trade_parameters": {
            "leverage": 20,
            "order_type": 1,
            "volume": 1,
            "open_type": 1
        }
    }
    with patch("models.utils.config.Config.__init__", return_value=None):
        config = Config("configs/config.yaml")
        config.threshold_buy = 0.7
        config.threshold_sell = 0.3
        config.trade_parameters = {
            "leverage": 20,
            "order_type": 1,
            "volume": 1,
            "open_type": 1
        }
        return config

@pytest.fixture
def mock_risk_manager():
    return AsyncMock(spec=RiskManager)

@pytest.fixture
def mock_model():
    model = AsyncMock(spec=torch.nn.Module)
    model.eval = AsyncMock()
    model.__call__ = AsyncMock(return_value=torch.sigmoid(torch.tensor([0.8])))
    return model

@pytest.fixture
def trading_agent(mock_config, mock_risk_manager, mock_model):
    scaler = MinMaxScaler()
    scaler.fit([[1,2,3], [4,5,6]])  # Dummy fit
    return TradingAgent(
        timeframes=["1m"],
        indicators=["price", "volume", "rsi", "macd", "fibonacci"],
        model=mock_model,
        config=mock_config,
        risk_manager=mock_risk_manager,
        scaler=scaler
    )

@pytest.mark.asyncio
async def test_trading_agent_make_decision_buy(trading_agent):
    # Mock market data indicating a buy signal
    market_data = {
        "1m": {
            "price": [35050.00],
            "volume": [100.5],
            "rsi": [50],
            "macd": [0.1],
            "fibonacci": [35000.00]
        }
    }

    with patch.object(trading_agent, '_execute_trade', new_callable=AsyncMock) as mock_execute_trade:
        with patch.object(trading_agent.risk_manager, 'check_risk', return_value=True):
            await trading_agent.make_decision(market_data)
            mock_execute_trade.assert_called_with('buy')

@pytest.mark.asyncio
async def test_trading_agent_make_decision_sell(trading_agent):
    # Mock market data indicating a sell signal
    trading_agent.model.return_value = torch.sigmoid(torch.tensor([-0.5]))
    market_data = {
        "1m": {
            "price": [35100.00],
            "volume": [101.0],
            "rsi": [60],
            "macd": [0.2],
            "fibonacci": [35050.00]
        }
    }

    with patch.object(trading_agent, '_execute_trade', new_callable=AsyncMock) as mock_execute_trade:
        with patch.object(trading_agent.risk_manager, 'check_risk', return_value=True):
            await trading_agent.make_decision(market_data)
            mock_execute_trade.assert_called_with('sell')

@pytest.mark.asyncio
async def test_trading_agent_make_decision_hold(trading_agent):
    # Mock market data indicating a hold signal
    trading_agent.model.return_value = torch.sigmoid(torch.tensor([0.5]))
    market_data = {
        "1m": {
            "price": [35150.00],
            "volume": [102.0],
            "rsi": [55],
            "macd": [0.15],
            "fibonacci": [35100.00]
        }
    }

    with patch.object(trading_agent, '_execute_trade', new_callable=AsyncMock) as mock_execute_trade:
        with patch.object(trading_agent.risk_manager, 'check_risk', return_value=True):
            await trading_agent.make_decision(market_data)
            mock_execute_trade.assert_not_called()

@pytest.mark.asyncio
async def test_trading_agent_risk_check_failed(trading_agent):
    # Mock market data indicating a buy signal but risk check fails
    trading_agent.model.return_value = torch.sigmoid(torch.tensor([0.8]))
    market_data = {
        "1m": {
            "price": [35200.00],
            "volume": [103.0],
            "rsi": [65],
            "macd": [0.25],
            "fibonacci": [35150.00]
        }
    }

    with patch.object(trading_agent.risk_manager, 'check_risk', return_value=False):
        with patch.object(trading_agent, '_execute_trade', new_callable=AsyncMock) as mock_execute_trade:
            await trading_agent.make_decision(market_data)
            mock_execute_trade.assert_not_called()

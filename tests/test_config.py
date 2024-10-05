# File: tests/test_config.py

import pytest
import os
from models.utils.config import Config
from unittest.mock import mock_open, patch
import yaml

def test_config_loading():
    mock_yaml = """
    symbol: BTC_USDT
    interval: Min1
    timeframes: ["1m", "5m", "15m", "1h", "4h"]
    indicators: ["price", "volume", "rsi", "macd", "fibonacci"]
    max_history_length: 1000
    lnn_model_path: models/lnn/lnn_model.pth
    lnn_hidden_size: 64
    lnn_training_epochs: 10
    training_history_length: 500
    lnn_learning_rate: 0.001
    threshold_buy: 0.7
    threshold_sell: 0.3
    risk_parameters:
      max_drawdown: 0.1
      max_position_size: 0.05
    trade_parameters:
      leverage: 20
      order_type: 1
      volume: 1
      open_type: 1
    agent_loop_delay: 1
    reconnect_delay: 5
    log_level: INFO
    """
    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        config = Config("configs/config.yaml")
        assert config.symbol == "BTC_USDT"
        assert config.interval == "Min1"
        assert config.timeframes == ["1m", "5m", "15m", "1h", "4h"]
        assert config.indicators == ["price", "volume", "rsi", "macd", "fibonacci"]
        assert config.max_history_length == 1000
        assert config.lnn_model_path == "models/lnn/lnn_model.pth"
        assert config.lnn_hidden_size == 64
        assert config.lnn_training_epochs == 10
        assert config.training_history_length == 500
        assert config.lnn_learning_rate == 0.001
        assert config.threshold_buy == 0.7
        assert config.threshold_sell == 0.3
        assert config.risk_parameters == {"max_drawdown": 0.1, "max_position_size": 0.05}
        assert config.trade_parameters == {
            "leverage": 20,
            "order_type": 1,
            "volume": 1,
            "open_type": 1
        }
        assert config.agent_loop_delay == 1
        assert config.reconnect_delay == 5
        assert config.log_level == "INFO"

def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        Config("configs/non_existent_config.yaml")

def test_config_invalid_yaml():
    invalid_yaml = "symbol: BTC_USDT\ninvalid_yaml: [unclosed"
    with patch("builtins.open", mock_open(read_data=invalid_yaml)):
        with pytest.raises(ValueError):
            Config("configs/config.yaml")

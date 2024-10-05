# File: models/utils/config.py

import yaml
import logging
from typing import List, Dict, Any


class Config:
    def __init__(self, config_path: str):
        try:
            with open(config_path, "r") as f:
                self.config: Dict[str, Any] = yaml.safe_load(f)

                # Trading Parameters
                self.symbol: str = self.config.get("symbol", "BTC_USDT")
                self.interval: str = self.config.get("interval", "Min1")
                self.timeframes: List[str] = self.config.get("timeframes", ["1m", "5m", "15m", "1h", "4h"])
                self.indicators: List[str] = self.config.get("indicators", ["price", "volume", "rsi", "macd", "fibonacci"])

                # GMN Parameters
                self.max_history_length: int = self.config.get("max_history_length", 1000)

                # LNN Parameters
                self.lnn_model_path: str = self.config.get("lnn_model_path", "models/lnn/lnn_model.pth")
                self.lnn_hidden_size: int = self.config.get("lnn_hidden_size", 64)
                self.lnn_training_epochs: int = self.config.get("lnn_training_epochs", 10)
                self.training_history_length: int = self.config.get("training_history_length", 500)
                self.lnn_learning_rate: float = self.config.get("lnn_learning_rate", 0.001)

                # Agent Parameters
                self.threshold_buy: float = self.config.get("threshold_buy", 0.7)
                self.threshold_sell: float = self.config.get("threshold_sell", 0.3)

                # Risk Management
                self.risk_parameters: Dict[str, Any] = self.config.get("risk_parameters", {})

                # Trade Execution
                self.trade_parameters: Dict[str, Any] = self.config.get("trade_parameters", {})

                # System
                self.agent_loop_delay: int = self.config.get("agent_loop_delay", 1)
                self.reconnect_delay: int = self.config.get("reconnect_delay", 5)
                self.log_level = self.config.get("log_level", "INFO")

        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config file: {e}")

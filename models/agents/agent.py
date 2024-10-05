# File: models/agents/agent.py

import asyncio
import logging
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from models.utils.config import Config
from models.utils.risk_management import RiskManager
import os
import aiohttp  # Ensure aiohttp is imported
import hashlib
import hmac
import time
import joblib

class TradingAgent:
    def __init__(
        self, 
        timeframes, 
        indicators, 
        model, 
        config: Config, 
        risk_manager: RiskManager, 
        scaler: MinMaxScaler
    ):
        self.timeframes = timeframes
        self.indicators = indicators
        self.model = model
        self.config = config
        self.threshold_buy = config.threshold_buy
        self.threshold_sell = config.threshold_sell
        self.risk_manager = risk_manager
        self.trade_parameters = config.trade_parameters
        self.leverage = self.trade_parameters.get("leverage", 20)
        self.position = None
        self.scaler = scaler
        self.api_key = os.getenv("MEXC_API_KEY")
        self.api_secret = os.getenv("MEXC_API_SECRET")
        self.base_url = 'https://contract.mexc.com/api/v1/'  # Update if necessary

        self.peak_portfolio_value = 1.0  # Initialize for drawdown calculations
        self.portfolio_value = 1.0  # Initialize portfolio value

        # Initialize aiohttp session
        self.session = aiohttp.ClientSession()

    async def make_decision(self, market_data):
        """Processes market data, makes predictions, and executes trades based on the model's output."""
        try:
            input_vector = self._prepare_input(market_data)
            if input_vector is None:
                logging.warning("Input vector is None. Skipping decision.")
                return

            # Model expects input in float32
            input_tensor = torch.tensor([input_vector], dtype=torch.float32)

            with torch.no_grad():
                self.model.eval()
                prediction = self.model(input_tensor)
                prediction = torch.sigmoid(prediction)  # Apply sigmoid to get probability
                prediction_value = prediction.item()

            logging.info(f"Model Prediction Probability: {prediction_value:.4f}")

            current_drawdown = self.calculate_current_drawdown()

            if self.risk_manager.check_risk(current_drawdown, self.position, market_data):
                if prediction_value >= self.threshold_buy and self.position != 'long':
                    await self._execute_trade('buy')
                    self.position = 'long'
                elif prediction_value <= self.threshold_sell and self.position != 'short':
                    await self._execute_trade('sell')
                    self.position = 'short'
                elif self.threshold_sell < prediction_value < self.threshold_buy:
                    if self.position is not None:
                        await self._execute_trade('close')
                        self.position = None
            else:
                logging.warning("Risk management check failed. Not executing trade.")

        except Exception as e:
            logging.error(f"Error in make_decision: {e}")

    def _prepare_input(self, market_data):
        """Prepares and scales the input vector for the LNN model."""
        input_vector = []
        for timeframe in self.timeframes:
            for indicator in self.indicators:
                data = market_data.get(timeframe, {}).get(indicator)
                if data is None or len(data) == 0:
                    logging.warning(f"Missing data for {timeframe} {indicator}. Skipping.")
                    return None

                if isinstance(data[-1], dict):
                    values = [v for v in data[-1].values() if isinstance(v, (int, float))]
                    input_vector.extend(values)
                else:
                    input_vector.append(data[-1])

        if not input_vector:
            logging.warning("Input vector is empty. No data available for making a decision.")
            return None

        input_vector = np.array([input_vector], dtype=np.float32)
        input_vector = self.scaler.transform(input_vector).astype(np.float32)
        return input_vector.flatten()

    async def _execute_trade(self, action, symbol="BTC_USDT"):
        """Executes a trade action (buy, sell, close) via the MEXC API using aiohttp."""
        try:
            timestamp = int(time.time() * 1000)
            params = {
                "symbol": symbol,
                "timestamp": timestamp
            }

            # Determine trade parameters based on action
            if action == 'buy':
                side = 'OPEN_LONG'
                quantity = self.trade_parameters.get("volume", 1)
            elif action == 'sell':
                side = 'OPEN_SHORT'
                quantity = self.trade_parameters.get("volume", 1)
            elif action == 'close':
                if self.position == 'long':
                    side = 'CLOSE_LONG'
                elif self.position == 'short':
                    side = 'CLOSE_SHORT'
                else:
                    logging.warning("No position to close.")
                    return
                quantity = self.trade_parameters.get("volume", 1)
            else:
                logging.warning(f"Invalid trade action: {action}")
                return

            # Set additional parameters
            params.update({
                "price": '',  # Empty for market orders
                "vol": quantity,
                "side": side,
                "type": self.trade_parameters.get("order_type", 1),  # 1: Market order
                "leverage": self.leverage,
                "openType": self.trade_parameters.get("open_type", 1),  # 1: Isolated margin
            })

            # Generate signature
            query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params)])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['sign'] = signature

            # Send request using aiohttp
            url = self.base_url + 'order/submit'
            async with self.session.post(url, params=params) as response:
                response_data = await response.json()
                if response.status == 200 and response_data.get("success", False):
                    logging.info(f"Successfully executed {action} order: {response_data}")
                    # Update portfolio value based on trade execution
                    self.update_portfolio(action, response_data)
                else:
                    logging.error(f"Failed to execute {action} order: {response_data}")

        except Exception as e:
            logging.error(f"Error executing trade: {e}")

    def calculate_current_drawdown(self):
        """Calculates the current drawdown based on portfolio value."""
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        logging.info(f"Current Drawdown: {drawdown:.4f}")
        return drawdown

    def update_portfolio(self, action: str, response_data: Dict):
        """
        Updates the portfolio value based on the executed trade.
        This is a placeholder function. You need to implement actual portfolio management logic.
        """
        # Example: Update portfolio based on the price and quantity
        try:
            price = float(response_data.get('data', {}).get('price', self.portfolio_value))
            quantity = float(response_data.get('data', {}).get('vol', 0))
            if action == 'buy':
                # Example logic: Increase portfolio value
                self.portfolio_value += price * quantity
            elif action == 'sell':
                # Example logic: Decrease portfolio value
                self.portfolio_value -= price * quantity
            elif action == 'close':
                # Example logic: Neutralize position
                pass
            logging.info(f"Portfolio updated after {action}: {self.portfolio_value}")
        except Exception as e:
            logging.error(f"Error updating portfolio: {e}")

    async def close(self):
        """Closes the aiohttp session."""
        await self.session.close()

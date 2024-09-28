# File: src/trading.py

import pandas as pd
from typing import Optional
from src.rewards import RewardFunction
from src.utils import get_logger

logger = get_logger()

class TradingExecutor:
    """
    Executes trading strategies based on model predictions.
    """

    def __init__(self, initial_balance: float = 10000.0, transaction_fee: float = 0.001, slippage: float = 0.0005):
        """
        Initializes the TradingExecutor.

        Args:
            initial_balance (float): Starting balance in USD.
            transaction_fee (float): Fee per trade.
            slippage (float): Slippage per trade.
        """
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage

    def execute_backtest(self, df: pd.DataFrame, features: pd.DataFrame, target: pd.Series, reward_function: RewardFunction) -> pd.DataFrame:
        """
        Executes a backtest of the trading strategy.

        Args:
            df (pd.DataFrame): Original DataFrame with market data.
            features (pd.DataFrame): Feature DataFrame used for predictions.
            target (pd.Series): Target variable.
            reward_function (RewardFunction): Reward function to calculate rewards.

        Returns:
            pd.DataFrame: Trade history.
        """
        balance = self.initial_balance
        position = 0
        trade_history = []

        for i in range(len(features) - 1):
            current_features = features.iloc[i].values
            current_price = df.iloc[i]['close']
            next_price = df.iloc[i + 1]['close']

            # Dummy prediction logic (replace with actual model predictions)
            # For example, buy if RSI < 30, sell if RSI > 70
            rsi = features.iloc[i]['RSI']
            if rsi < 30 and balance > 0:
                # Buy
                amount_to_buy = balance * 0.1  # Buy 10% of balance
                position += amount_to_buy / current_price
                balance -= amount_to_buy * (1 + self.transaction_fee + self.slippage)
                action = 'Buy'
            elif rsi > 70 and position > 0:
                # Sell
                proceeds = position * current_price
                balance += proceeds * (1 - self.transaction_fee - self.slippage)
                position = 0
                action = 'Sell'
            else:
                action = 'Hold'

            # Calculate portfolio value
            portfolio_value = balance + position * current_price

            # Calculate reward
            reward = reward_function.calculate_reward(
                action=action,
                current_price=current_price,
                next_price=next_price,
                portfolio_value=portfolio_value
            )

            # Record trade
            trade_history.append({
                'step': i,
                'action': action,
                'balance': balance,
                'position': position,
                'portfolio_value': portfolio_value,
                'reward': reward
            })

        trade_history_df = pd.DataFrame(trade_history)
        logger.info("Backtest completed.")
        return trade_history_df

    def execute_live_trading(self, df: pd.DataFrame, features: pd.DataFrame, model, reward_function: RewardFunction) -> pd.DataFrame:
        """
        Executes live trading using the trained model.

        Args:
            df (pd.DataFrame): DataFrame with live market data.
            features (pd.DataFrame): Feature DataFrame used for predictions.
            model: Trained trading model.
            reward_function (RewardFunction): Reward function to calculate rewards.

        Returns:
            pd.DataFrame: Trade history.
        """
        balance = self.initial_balance
        position = 0
        trade_history = []

        model.eval()

        with torch.no_grad():
            for i in range(len(features) - 1):
                current_features = torch.tensor(features.iloc[i].values, dtype=torch.float32).unsqueeze(0)
                current_price = df.iloc[i]['close']
                next_price = df.iloc[i + 1]['close']

                # Get model prediction
                outputs = model(current_features)
                action = torch.argmax(outputs, dim=1).item()  # 0: Hold, 1: Buy, 2: Sell

                # Map action to string
                action_str = {0: 'Hold', 1: 'Buy', 2: 'Sell'}[action]

                if action_str == 'Buy' and balance > 0:
                    # Buy
                    amount_to_buy = balance * 0.1  # Buy 10% of balance
                    position += amount_to_buy / current_price
                    balance -= amount_to_buy * (1 + self.transaction_fee + self.slippage)
                elif action_str == 'Sell' and position > 0:
                    # Sell
                    proceeds = position * current_price
                    balance += proceeds * (1 - self.transaction_fee - self.slippage)
                    position = 0

                # Calculate portfolio value
                portfolio_value = balance + position * current_price

                # Calculate reward
                reward = reward_function.calculate_reward(
                    action=action,
                    current_price=current_price,
                    next_price=next_price,
                    portfolio_value=portfolio_value
                )

                # Record trade
                trade_history.append({
                    'step': i,
                    'action': action_str,
                    'balance': balance,
                    'position': position,
                    'portfolio_value': portfolio_value,
                    'reward': reward
                })

        trade_history_df = pd.DataFrame(trade_history)
        logger.info("Live trading execution completed.")
        return trade_history_df

# models/evaluator.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List
from utils.utils import get_logger

logger = get_logger()

class Evaluator:
    """
    Evaluates trading strategies based on various performance metrics.
    """

    def __init__(self, trade_history: pd.DataFrame):
        """
        Initializes the evaluator with trade history data.

        Args:
            trade_history (pd.DataFrame): DataFrame containing trade details.
        """
        self.trade_history = trade_history

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculates the Sharpe Ratio of the trading strategy.

        Args:
            risk_free_rate (float): The risk-free rate. Defaults to 0.0.

        Returns:
            float: Sharpe Ratio.
        """
        returns = self.trade_history['returns']
        excess_returns = returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def calculate_max_drawdown(self) -> float:
        """
        Calculates the Maximum Drawdown of the trading strategy.

        Returns:
            float: Maximum Drawdown.
        """
        cumulative_returns = (1 + self.trade_history['returns']).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_total_return(self) -> float:
        """
        Calculates the Total Return of the trading strategy.

        Returns:
            float: Total Return.
        """
        total_return = (self.trade_history['portfolio_value'].iloc[-1] / self.trade_history['portfolio_value'].iloc[0]) - 1
        return total_return

    def plot_equity_curve(self):
        """
        Plots the equity curve of the trading strategy.
        """
        cumulative_returns = (1 + self.trade_history['returns']).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns, label='Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative Returns')
        plt.title('Equity Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_drawdown(self):
        """
        Plots the drawdown over time.
        """
        cumulative_returns = (1 + self.trade_history['returns']).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown, label='Drawdown', color='red')
        plt.xlabel('Trade Number')
        plt.ylabel('Drawdown')
        plt.title('Drawdown Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def summary(self):
        """
        Prints a summary of key performance metrics.
        """
        sharpe = self.calculate_sharpe_ratio()
        max_dd = self.calculate_max_drawdown()
        total_ret = self.calculate_total_return()

        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Maximum Drawdown: {max_dd:.2%}")
        print(f"Total Return: {total_ret:.2%}")

# Example usage
# evaluator = Evaluator(trade_history=trade_history_df)
# evaluator.summary()
# evaluator.plot_equity_curve()
# evaluator.plot_drawdown()

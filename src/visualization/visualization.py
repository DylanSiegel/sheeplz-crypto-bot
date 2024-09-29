# File: src/visualization/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from src.utils.utils import get_logger

logger = get_logger(__name__)

class Visualization:
    """
    Provides various visualization methods for trading data, 
    performance, and training progress.
    """

    def __init__(self):
        pass

    def plot_price_chart(self, df: pd.DataFrame, title: str = "Price Chart"):
        """
        Creates an interactive candlestick chart with optional technical indicators.

        Args:
            df (pd.DataFrame): DataFrame containing price data ('open', 'high', 'low', 'close', 'volume').
            title (str, optional): Title of the chart. Defaults to "Price Chart".
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                           subplot_titles=(title, "Volume"))

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="Price"), row=1, col=1)

        # Add indicators (example: SMA and RSI)
        if 'SMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'), row=1, col=1)
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)

        # Volume bar chart
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'), row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, height=800)
        st.plotly_chart(fig)

    def plot_performance_metrics(self, performance_data: Dict):
        """
        Visualizes key performance metrics such as equity curve, Sharpe ratio, 
        and maximum drawdown.

        Args:
            performance_data (Dict): Dictionary containing performance data.
                Example: {'portfolio_value': [], 'returns': [], ...}
        """
        df = pd.DataFrame(performance_data)

        # Equity Curve
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=df.index, y=df['portfolio_value'], mode='lines', name='Equity Curve'))
        fig_equity.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Portfolio Value")
        st.plotly_chart(fig_equity)

        # Sharpe Ratio (Display as a single value)
        sharpe_ratio = self.calculate_sharpe_ratio(df['returns'])
        st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

        # Maximum Drawdown (Display as a single value)
        max_drawdown = self.calculate_max_drawdown(df['portfolio_value'])
        st.write(f"**Maximum Drawdown:** {max_drawdown:.2%}")

    def display_order_history(self, order_history: List[Dict]):
        """
        Displays the order history in a tabular format.

        Args:
            order_history (List[Dict]): List of order dictionaries.
        """
        df = pd.DataFrame(order_history)
        st.table(df)

    def plot_training_progress(self, training_data: List[Dict]):
        """
        Visualizes the training progress, including reward/loss curves, 
        actions over time, etc.

        Args:
            training_data (List[Dict]): List of dictionaries containing training data 
                for each episode.
        """
        df = pd.DataFrame(training_data)

        # Reward/Loss Curve
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(x=df['episode'], y=df['reward'], mode='lines', name='Reward'))
        if 'loss' in df.columns:
            fig_reward.add_trace(go.Scatter(x=df['episode'], y=df['loss'], mode='lines', name='Loss'))
        fig_reward.update_layout(title="Reward/Loss Curve", xaxis_title="Episode", yaxis_title="Reward/Loss")
        st.plotly_chart(fig_reward)

        # Actions Over Time (Example with bar chart)
        if 'actions' in df.columns:
            action_counts = df['actions'].explode().value_counts()
            fig_actions = go.Figure(data=[go.Bar(x=action_counts.index, y=action_counts.values)])
            fig_actions.update_layout(title="Action Distribution", xaxis_title="Action", yaxis_title="Count")
            st.plotly_chart(fig_actions)

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculates the Sharpe Ratio.
        """
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """
        Calculates the Maximum Drawdown.
        """
        peak = portfolio_value.expanding(min_periods=1).max()
        drawdown = (portfolio_value - peak) / peak
        return drawdown.min()
import numpy as np
import pandas as pd
import ta
from typing import List, Union
from dataclasses import dataclass

@dataclass
class IndicatorConfig:
    """Configuration for technical indicator calculations.

    Attributes:
        rsi_period (int): The period for Relative Strength Index (RSI) calculation.
        macd_fast (int): The fast period for Moving Average Convergence Divergence (MACD) calculation.
        macd_slow (int): The slow period for MACD calculation.
        macd_signal (int): The signal period for MACD calculation.
        bb_period (int): The period for Bollinger Bands calculation.
        bb_std (int): The standard deviation multiplier for Bollinger Bands calculation.

    """
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2

class TechnicalIndicators:
    """Calculates various technical indicators from price data.

    This class uses the `TA-Lib` library to efficiently compute several common technical indicators.

    Args:
        config (IndicatorConfig, optional): Configuration object specifying parameters for indicator calculations. 
                                            Defaults to `IndicatorConfig()`.

    Methods:
        calculate_all(prices: Union[List[float], np.ndarray]) -> dict: Calculates all configured indicators.
        calculate_rsi(df: pd.DataFrame) -> float: Calculates the Relative Strength Index (RSI).
        calculate_macd(df: pd.DataFrame) -> dict: Calculates the Moving Average Convergence Divergence (MACD) and its signal and difference.
        calculate_bollinger_bands(df: pd.DataFrame) -> dict: Calculates the Bollinger Bands (upper, middle, lower).
        calculate_volatility(df: pd.DataFrame, window: int = 20) -> float: Calculates annualized volatility.

    """
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()

    def calculate_all(self, prices: Union[List[float], np.ndarray]) -> dict:
        """Calculates all configured technical indicators.

        Args:
            prices (Union[List[float], np.ndarray]): A list or NumPy array of closing prices.

        Returns:
            dict: A dictionary containing all calculated indicators.

        """
        df = pd.DataFrame({'close': prices})
        indicators = {}
        indicators['rsi'] = self.calculate_rsi(df)
        macd_data = self.calculate_macd(df)
        indicators.update(macd_data)
        bb_data = self.calculate_bollinger_bands(df)
        indicators.update(bb_data)
        indicators['volatility'] = self.calculate_volatility(df)
        return indicators

    def calculate_rsi(self, df: pd.DataFrame) -> float:
        """Calculates the Relative Strength Index (RSI).

        Args:
            df (pd.DataFrame): Pandas DataFrame with a 'close' column containing closing prices.

        Returns:
            float: The RSI value.

        """
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=self.config.rsi_period)
        return rsi.rsi().iloc[-1]

    def calculate_macd(self, df: pd.DataFrame) -> dict:
        """Calculates the Moving Average Convergence Divergence (MACD) and its signal and difference.

        Args:
            df (pd.DataFrame): Pandas DataFrame with a 'close' column containing closing prices.

        Returns:
            dict: A dictionary containing the MACD, MACD signal, and MACD difference.

        """
        macd = ta.trend.MACD(close=df['close'], window_slow=self.config.macd_slow, window_fast=self.config.macd_fast, window_sign=self.config.macd_signal)
        return {'macd': macd.macd().iloc[-1], 'macd_signal': macd.macd_signal().iloc[-1], 'macd_diff': macd.macd_diff().iloc[-1]}

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> dict:
        """Calculates the Bollinger Bands (upper, middle, lower).

        Args:
            df (pd.DataFrame): Pandas DataFrame with a 'close' column containing closing prices.

        Returns:
            dict: A dictionary containing the upper, middle, and lower Bollinger Bands.

        """
        bb = ta.volatility.BollingerBands(close=df['close'], window=self.config.bb_period, window_dev=self.config.bb_std)
        return {'bb_high': bb.bollinger_hband().iloc[-1], 'bb_mid': bb.bollinger_mavg().iloc[-1], 'bb_low': bb.bollinger_lband().iloc[-1]}

    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calculates annualized volatility.

        Args:
            df (pd.DataFrame): Pandas DataFrame with a 'close' column containing closing prices.
            window (int, optional): The rolling window size for volatility calculation. Defaults to 20.

        Returns:
            float: The annualized volatility.

        """
        returns = df['close'].pct_change()
        return returns.std() * np.sqrt(252)  # Annualized volatility
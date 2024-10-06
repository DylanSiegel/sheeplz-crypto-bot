import pandas as pd
from finta import TA
from error_handler import ErrorHandler
from typing import Dict, Any, List

class IndicatorCalculator:
    def __init__(self, error_handler: ErrorHandler):
        """
        Initializes the IndicatorCalculator with an error handler.

        Args:
            error_handler (ErrorHandler): Instance to handle errors during calculations.
        """
        self.error_handler = error_handler

    def calculate_indicators(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Calculates indicators for all timeframes.

        Args:
            symbol (str): The trading symbol (e.g., 'BTC_USDT').
            data (Dict[str, pd.DataFrame]): Dictionary where keys are timeframes and values are DataFrames.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of indicators keyed by timeframe.
        """
        indicators = {}
        for timeframe, df in data.items():
            try:
                indicators[timeframe] = {}
                indicators[timeframe]['rsi'] = self.calculate_rsi(df)
                indicators[timeframe]['macd'] = self.calculate_macd(df)
                indicators[timeframe]['fibonacci'] = self.calculate_fibonacci(df)
            except Exception as e:
                self.error_handler.handle_error(
                    f"Error calculating indicators for {symbol} {timeframe}: {e}",
                    exc_info=True
                )
        return indicators

    def calculate_rsi(self, df: pd.DataFrame) -> List[float]:
        """
        Calculates the RSI (Relative Strength Index).

        Args:
            df (pd.DataFrame): DataFrame containing OHLC data.

        Returns:
            List[float]: RSI values.
        """
        try:
            rsi_values = TA.RSI(df).fillna(0).tolist()
            return rsi_values
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating RSI: {e}", exc_info=True)
            return []

    def calculate_macd(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Calculates the MACD (Moving Average Convergence Divergence).

        Args:
            df (pd.DataFrame): DataFrame containing OHLC data.

        Returns:
            Dict[str, List[float]]: MACD values, signal line, and histogram.
        """
        try:
            macd_values = TA.MACD(df).fillna(0)
            return {
                'macd': macd_values['MACD'].tolist(),
                'macd_signal': macd_values['SIGNAL'].tolist(),
                'macd_hist': macd_values['HISTOGRAM'].tolist()
            }
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating MACD: {e}", exc_info=True)
            return {'macd': [], 'macd_signal': [], 'macd_hist': []}

    def calculate_fibonacci(self, df: pd.DataFrame) -> List[float]:
        """
        Calculates Fibonacci Retracement levels manually.

        Args:
            df (pd.DataFrame): DataFrame containing OHLC data.

        Returns:
            List[float]: Fibonacci retracement levels.
        """
        try:
            high = df['high'].max()
            low = df['low'].min()
            fib_levels = [high - ((high - low) * ratio) for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]
            return fib_levels
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating Fibonacci Retracement: {e}", exc_info=True)
            return []

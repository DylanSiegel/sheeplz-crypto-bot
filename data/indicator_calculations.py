import pandas as pd
import pandas_ta as ta
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
                indicators[timeframe]['rsi'] = self.calculate_rsi(df['close'])  # Use 'close' column
                indicators[timeframe]['macd'] = self.calculate_macd(df['close'])  # Use 'close' column
                indicators[timeframe]['fibonacci'] = self.calculate_fibonacci(df['close'])  # Use 'close' column
                # ... add other indicators as needed
            except Exception as e:
                self.error_handler.handle_error(
                    f"Error calculating indicators for {symbol} {timeframe}: {e}",
                    exc_info=True
                )
        return indicators

    def calculate_rsi(self, price_series: pd.Series) -> List[float]:
        """
        Calculates the RSI (Relative Strength Index).

        Args:
            price_series (pd.Series): Series of closing prices.

        Returns:
            List[float]: RSI values.
        """
        try:
            rsi_values = ta.rsi(price_series, length=14)
            return rsi_values.fillna(0).tolist()
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating RSI: {e}", exc_info=True)
            return []

    def calculate_macd(self, price_series: pd.Series) -> Dict[str, List[float]]:
        """
        Calculates the MACD (Moving Average Convergence Divergence).

        Args:
            price_series (pd.Series): Series of closing prices.

        Returns:
            Dict[str, List[float]]: MACD values, signal line, and histogram.
        """
        try:
            macd = ta.macd(price_series, fast=12, slow=26, signal=9)
            return {
                'macd': macd['MACD_12_26_9'].fillna(0).tolist(),
                'macd_signal': macd['MACDs_12_26_9'].fillna(0).tolist(),
                'macd_hist': macd['MACDh_12_26_9'].fillna(0).tolist()
            }
        except KeyError as e:
            self.error_handler.handle_error(f"Error calculating MACD: Missing key - {e}", exc_info=True)
            return {'macd': [], 'macd_signal': [], 'macd_hist': []}
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating MACD: {e}", exc_info=True)
            return {'macd': [], 'macd_signal': [], 'macd_hist': []}

    def calculate_fibonacci(self, price_series: pd.Series) -> List[float]:
        """
        Calculates Fibonacci Retracement levels.

        Args:
            price_series (pd.Series): Series of closing prices.

        Returns:
            List[float]: Fibonacci Retracement values.
        """
        try:
            fib_values = []
            for i in range(len(price_series)):
                if i < 14:
                    fib_values.append(float('nan'))
                else:
                    recent_prices = price_series.iloc[i - 14:i]
                    high = recent_prices.max()
                    low = recent_prices.min()
                    close = recent_prices.iloc[-1]
                    diff = high - low
                    fib = (close - low) / diff if diff != 0 else float('nan')
                    fib_values.append(fib)
            return fib_values
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating Fibonacci Retracement: {e}", exc_info=True)
            return []

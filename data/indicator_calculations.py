import pandas as pd
from finta import TA
from .error_handler import ErrorHandler
from typing import Dict, Any, List
import concurrent.futures

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
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._calculate_for_timeframe, symbol, timeframe, df): timeframe for timeframe, df in data.items()}
                for future in concurrent.futures.as_completed(futures):
                    timeframe = futures[future]
                    try:
                        indicators[timeframe] = future.result()
                    except Exception as e:
                        self.error_handler.handle_error(f"Error calculating indicators for {symbol} {timeframe}: {e}", exc_info=True)
        except Exception as e:
            self.error_handler.handle_error(f"Error in calculate_indicators: {e}", exc_info=True)
        return indicators

    def _calculate_for_timeframe(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Helper function to calculate indicators for a specific timeframe.
        Args:
            symbol (str): The trading symbol (e.g., 'BTC_USDT').
            timeframe (str): The timeframe being processed.
            df (pd.DataFrame): DataFrame containing OHLC data for the timeframe.
        Returns:
            Dict[str, Any]: Calculated indicators for the timeframe.
        """
        indicators = {}
        try:
            # Ensure dataframe has the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns in {timeframe} data for {symbol}: {df.columns}")

            indicators['rsi'] = self.calculate_rsi(df)
            indicators['macd'] = self.calculate_macd(df)
            indicators['fibonacci'] = self.calculate_fibonacci(df)
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating indicators for {symbol} {timeframe}: {e}", exc_info=True)
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

    def calculate_fibonacci(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates Fibonacci Retracement levels manually.
        Args:
            df (pd.DataFrame): DataFrame containing OHLC data.
        Returns:
            Dict[str, float]: Fibonacci retracement levels.
        """
        try:
            high = df['high'].max()
            low = df['low'].min()
            diff = high - low
            fib_levels = {
                "23.6%": high - 0.236 * diff,
                "38.2%": high - 0.382 * diff,
                "50.0%": high - 0.5 * diff,
                "61.8%": high - 0.618 * diff,
                "78.6%": high - 0.786 * diff
            }
            return fib_levels
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating Fibonacci Retracement: {e}", exc_info=True)
            return {}

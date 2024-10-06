import pandas as pd
import pandas_ta as ta
from error_handler import ErrorHandler

class IndicatorCalculator:
    def __init__(self, data_storage):
        self.data_storage = data_storage
        self.error_handler = ErrorHandler()  # Initialize the error handler

    def calculate_indicators(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """
        Calculates technical indicators based on the provided data.

        Args:
            symbol (str): The trading symbol (e.g., 'BTCUSDT').
            timeframe (str): The timeframe (e.g., '1m', '5m').
            data (pd.DataFrame): The market data, containing 'close' prices.

        Returns:
            dict: A dictionary containing calculated indicators.
        """
        try:
            indicators = {}

            # Calculate RSI
            rsi_values = self.calculate_rsi(data['close'])
            indicators['rsi'] = rsi_values

            # Calculate MACD
            macd_values, macd_signal, macd_hist = self.calculate_macd(data['close'])
            indicators['macd'] = macd_values
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist

            # Calculate Bollinger Bands
            bbands_values = self.calculate_bbands(data['close'])
            indicators['bbands'] = bbands_values

            # Calculate Fibonacci Retracement
            fib_values = self.calculate_fibonacci(data['close'])
            indicators['fibonacci'] = fib_values

            return indicators

        except KeyError as e:
            # Log specific key errors (e.g., missing columns in data)
            self.error_handler.handle_error(f"KeyError: Missing data column - {e}", exc_info=True)
        except Exception as e:
            # Handle all other exceptions
            self.error_handler.handle_error(f"Error calculating indicators: {e}", exc_info=True)

        # Return an empty dictionary if the calculation fails
        return {}

    def calculate_rsi(self, close_prices: pd.Series):
        """Calculates the RSI (Relative Strength Index)."""
        try:
            rsi_values = ta.rsi(close_prices, length=14)
            return rsi_values.tolist()
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating RSI: {e}", exc_info=True)
            return []

    def calculate_macd(self, close_prices: pd.Series):
        """Calculates the MACD (Moving Average Convergence Divergence)."""
        try:
            macd = ta.macd(close_prices, fast=12, slow=26, signal=9)
            macd_values = macd['MACD_12_26_9'].tolist()
            macd_signal = macd['MACDs_12_26_9'].tolist()
            macd_hist = macd['MACDh_12_26_9'].tolist()
            return macd_values, macd_signal, macd_hist
        except KeyError as e:
            self.error_handler.handle_error(f"Error calculating MACD: Missing key - {e}", exc_info=True)
            return [], [], []
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating MACD: {e}", exc_info=True)
            return [], [], []

    def calculate_bbands(self, close_prices: pd.Series):
        """Calculates Bollinger Bands."""
        try:
            bbands = ta.bbands(close_prices, length=20, std=2)
            return bbands['BBM_20_2'].tolist()
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating Bollinger Bands: {e}", exc_info=True)
            return []

    def calculate_fibonacci(self, close_prices: pd.Series):
        """Calculates Fibonacci Retracement levels."""
        try:
            fib_values = []
            for i in range(len(close_prices)):
                if i < 14:
                    fib_values.append(float('nan'))
                else:
                    recent_prices = close_prices.iloc[i - 14:i]
                    high = max(recent_prices)
                    low = min(recent_prices)
                    close = recent_prices.iloc[-1]
                    diff = high - low
                    fib_values.append((close - low) / diff if diff != 0 else float('nan'))
            return fib_values
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating Fibonacci Retracement: {e}", exc_info=True)
            return []

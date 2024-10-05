from typing import Callable, Optional, Dict, Union
import numpy as np
import talib
from logger import logger

class IndicatorFactory:
    @staticmethod
    def create_rsi(timeperiod: int = 14) -> Callable[[np.ndarray], Optional[float]]:
        return lambda prices: talib.RSI(prices, timeperiod=timeperiod)[-1] if len(prices) >= timeperiod else None

    @staticmethod
    def create_macd(fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Callable[[np.ndarray], Optional[float]]:
        def macd_func(prices: np.ndarray) -> Optional[float]:
            try:
                macd, _, _ = talib.MACD(prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
                latest_macd = macd[-1]
                return float(latest_macd) if not np.isnan(latest_macd) else None
            except Exception as e:
                logger.error(f"Error calculating MACD: {e}")
                return None
        return macd_func

    @staticmethod
    def create_ema(timeperiod: int = 30) -> Callable[[np.ndarray], Optional[float]]:
        return lambda prices: talib.EMA(prices, timeperiod=timeperiod)[-1] if len(prices) >= timeperiod else None

    @staticmethod
    def create_fibonacci() -> Callable[[np.ndarray], Optional[Dict[str, Union[float, bool]]]]:
        def fibonacci_func(prices: np.ndarray, lookback: int = 14) -> Optional[Dict[str, Union[float, bool]]]:
            try:
                if len(prices) < lookback:
                    logger.warning("Not enough data to calculate Fibonacci retracement.")
                    return None
                recent_prices = prices[-lookback:]
                high = np.max(recent_prices)
                low = np.min(recent_prices)
                close = recent_prices[-1]
                diff = high - low
                if diff == 0:
                    logger.warning("High and low prices are the same; cannot calculate Fibonacci levels.")
                    return None

                levels = {
                    "23.6%": high - 0.236 * diff,
                    "38.2%": high - 0.382 * diff,
                    "50%": high - 0.5 * diff,
                    "61.8%": high - 0.618 * diff,
                    "78.6%": high - 0.786 * diff,
                    "100%": low,
                }
                closest_level_key = min(levels, key=lambda k: abs(levels[k] - close))
                return {k: {"value": v, "is_closest": k == closest_level_key} for k, v in levels.items()}
            except Exception as e:
                logger.error(f"Error calculating Fibonacci retracement: {e}")
                return None
        return fibonacci_func
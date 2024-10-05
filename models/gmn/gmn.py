import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union, Callable, cast
import numpy as np
import talib  # type: ignore  # Suppress type errors for TA-Lib
import logging
from configparser import ConfigParser
from pydantic import BaseModel, ValidationError, validator

logger = logging.getLogger(__name__)

# Load configuration from external file (config.ini)
config = ConfigParser()
config.read("config.ini")  # Ensure a config.ini file exists in the same directory

# Default configuration values
DEFAULT_TIMEFRAMES = config.get("DEFAULT", "timeframes", fallback="1m,5m").split(",")  # Example: ['1m', '5m']
DEFAULT_INDICATORS = config.get("DEFAULT", "indicators", fallback="price,volume,rsi,macd,fibonacci").split(",")  # Example: ['price', 'volume', 'rsi', 'macd', 'fibonacci']
MAX_HISTORY = config.getint("DEFAULT", "max_history", fallback=1000)
EXECUTOR_WORKERS = config.getint("DEFAULT", "executor_workers", fallback=5)


class CryptoGMNError(Exception):
    """Base exception for CryptoGMN."""
    pass


class IndicatorCalculationError(CryptoGMNError):
    """Exception raised when an indicator calculation fails."""
    pass


class DataModel(BaseModel):
    """Pydantic model for validating incoming data."""
    c: float  # Close price
    v: Optional[float] = 0.0  # Volume

    @validator('c', 'v', pre=True)
    def validate_numeric(cls, v):
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Value '{v}' is not a valid float.")


class CryptoGMN:
    """Manages cryptocurrency market data and technical indicators."""

    def __init__(self, timeframes: List[str] = DEFAULT_TIMEFRAMES,  # Default values
                 indicators: List[str] = DEFAULT_INDICATORS,
                 max_history_length: int = MAX_HISTORY,
                 executor_workers: int = EXECUTOR_WORKERS) -> None:
        """
        Initializes the CryptoGMN instance.

        :param timeframes: List of timeframes (e.g., ['1m', '5m', '1h'])
        :param indicators: List of indicators (e.g., ['price', 'volume', 'rsi', 'macd', 'fibonacci'])
        :param max_history_length: Maximum number of data points to store per indicator
        :param executor_workers: Number of worker threads for indicator calculations
        """
        self.timeframes = timeframes
        self.indicators = indicators
        self.max_history_length = max_history_length
        self.market_data: Dict[str, Dict[str, deque]] = {
            timeframe: {indicator: deque(maxlen=max_history_length) for indicator in indicators}
            for timeframe in timeframes
        }
        self.locks: Dict[str, asyncio.Lock] = {timeframe: asyncio.Lock() for timeframe in timeframes}
        self.executor: ThreadPoolExecutor  # Explicitly type hint
        # Asynchronous context manager for Executor
        self._executor_lock = asyncio.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None

    async def __aenter__(self):
        async with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.executor_workers)  # Use the config value here
            self.executor = cast(ThreadPoolExecutor, self._executor)  # Cast to make mypy happy.
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def update_graph(self, new_data_items: List[Dict[str, Any]]) -> None:
        """Updates market data with new items concurrently."""
        await asyncio.gather(*(self._update_single_data(data) for data in new_data_items))  # More concise

    async def _update_single_data(self, new_data: Dict[str, Any]) -> None:
        """Updates data for a single new item across all timeframes."""
        try:
            validated_data = DataModel(**new_data)
            price = validated_data.c
            volume = validated_data.v
        except ValidationError as e:
            logger.error(f"Data validation error: {e}. Data: {new_data}")
            return

        await asyncio.gather(*(self._update_timeframe(tf, price, volume) for tf in self.timeframes))

    async def _update_timeframe(self, timeframe: str, price: float, volume: float) -> None:
        """Updates data for a specific timeframe, including batching for high-frequency data."""
        async with self.locks[timeframe]:
            data = self.market_data[timeframe]
            data['price'].append(price)
            data['volume'].append(volume)

            prices_array = np.array(data['price'], dtype=np.float64)

            if len(prices_array) >= 14:
                await self._calculate_indicators(timeframe, prices_array)

    async def _calculate_indicators(self, timeframe: str, prices_array: np.ndarray):
        """Calculates all indicators for a timeframe concurrently."""
        tasks = []
        if 'rsi' in self.indicators:
            tasks.append(self._calculate_and_append(timeframe, 'rsi', talib.RSI, prices_array, {'timeperiod': 14}))
        if 'macd' in self.indicators:
            tasks.append(self._calculate_and_append(
                timeframe, 'macd',
                lambda prices: self._macd_wrapper(prices),  # Lambda for MACD extraction
                prices_array
            ))
        if 'fibonacci' in self.indicators:
            tasks.append(self._calculate_and_append_fibonacci(timeframe, prices_array))
        # Add other indicators similarly if needed
        if tasks:
            await asyncio.gather(*tasks)

    async def _calculate_and_append(self, timeframe: str, indicator: str,
                                    indicator_func: Callable, prices: np.ndarray,
                                    kwargs: Dict[str, Any] = {}) -> None:
        """
        Calculates an indicator and appends the result to the market data.

        :param timeframe: The timeframe for which to calculate the indicator
        :param indicator: The name of the indicator
        :param indicator_func: The TA-Lib function to calculate the indicator
        :param prices: NumPy array of prices
        :param kwargs: Additional keyword arguments for the indicator function
        """
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self.executor, indicator_func, prices, **kwargs  # Simplify executor call
            )
        except Exception as e:
            logger.error(f"Error running indicator '{indicator}' for timeframe '{timeframe}': {e}")
            return

        if result is not None:
            try:
                latest = result[-1]
                if isinstance(latest, np.ndarray):
                    latest = latest.item()  # Convert numpy scalar to Python float
                if np.isnan(latest):
                    logger.warning(f"{indicator.upper()} returned NaN or empty result for timeframe {timeframe}")
                    return
                self.market_data[timeframe][indicator].append(float(latest))
            except IndexError:
                logger.warning(f"{indicator.upper()} returned NaN or empty result for timeframe {timeframe}")

    async def _calculate_and_append_fibonacci(self, timeframe: str, prices: np.ndarray, lookback: int = 14) -> None:
        """Calculates and appends Fibonacci levels."""
        loop = asyncio.get_running_loop()
        try:
            fibonacci_levels = await loop.run_in_executor(
                self.executor, self._calculate_fibonacci, prices, lookback
            )
        except Exception as e:
            logger.error(f"Error calculating Fibonacci for timeframe '{timeframe}': {e}")
            return

        if fibonacci_levels:
            self.market_data[timeframe]['fibonacci'].append(fibonacci_levels)

    def _macd_wrapper(self, prices: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Optional[float]:
        """Wrapper for talib.MACD to return only the MACD line."""
        try:
            macd, _, _ = talib.MACD(prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
            if macd is None or len(macd) == 0:
                logger.warning(f"MACD calculation returned no data for timeframe.")
                return None
            latest_macd = macd[-1]
            if np.isnan(latest_macd):
                logger.warning(f"MACD returned NaN for timeframe.")
                return None
            return float(latest_macd)
        except Exception as e:
            logger.error(f"Error in _macd_wrapper: {e}")
            return None

    def _calculate_fibonacci(self, prices: np.ndarray, lookback: int) -> Optional[Dict[str, Union[float, bool]]]:
        """Calculates Fibonacci retracement levels and indicates the closest level."""
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
                return None  # Avoid division by zero

            levels = {
                "23.6%": high - 0.236 * diff,
                "38.2%": high - 0.382 * diff,
                "50%": high - 0.5 * diff,
                "61.8%": high - 0.618 * diff,
                "78.6%": high - 0.786 * diff,
                "100%": low,
            }
            closest_level_key = min(
                levels.keys(), key=lambda k: abs(levels[k] - close)
            )
            for k in levels:
                levels[k] = {"value": levels[k], "is_closest": k == closest_level_key}  # Indicate closest

            return levels
        except Exception as e:
            logger.error(f"Error calculating Fibonacci retracement: {e}")
            return None

    def get_data(self, timeframe: str, indicator: str) -> Optional[List[Union[float, Dict[str, Union[float, bool]]]]]:
        """Retrieves the latest data for a specific timeframe and indicator."""
        try:
            return list(self.market_data[timeframe][indicator])
        except KeyError:
            logger.warning(
                f"No data found for timeframe '{timeframe}' and indicator '{indicator}'."
            )
            return None

    def get_all_data(self) -> Dict[str, Dict[str, List[Union[float, Dict[str, Union[float, bool]]]]]]:
        """Retrieves all market data across all timeframes and indicators."""
        return {
            timeframe: {
                indicator: list(data)
                for indicator, data in indicators.items()
            }
            for timeframe, indicators in self.market_data.items()
        }

    async def shutdown(self) -> None:
        """Shuts down the executor gracefully."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None  # Important for proper cleanup
        logger.info("CryptoGMN has been shut down gracefully.")

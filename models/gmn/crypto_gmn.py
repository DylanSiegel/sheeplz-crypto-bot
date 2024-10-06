import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class CryptoGMNError(Exception):
    """Base exception for CryptoGMN."""
    pass

class DataValidationError(CryptoGMNError):
    """Exception raised when data validation fails."""
    pass

class DataModel(BaseModel):
    c: float
    v: Optional[float] = 0.0

    class Config:
        arbitrary_types_allowed = True

class CryptoGMN:
    def __init__(self, timeframes: List[str], max_history_length: int, executor_workers: int):
        self.timeframes = timeframes
        self.max_history_length = max_history_length
        self.market_data: Dict[str, Dict[str, deque]] = {
            timeframe: {'price': deque(maxlen=max_history_length), 'volume': deque(maxlen=max_history_length)}
            for timeframe in timeframes
        }
        self.locks: Dict[str, asyncio.Lock] = {timeframe: asyncio.Lock() for timeframe in timeframes}
        self._executor_lock = asyncio.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self.executor_workers = executor_workers

    async def __aenter__(self):
        async with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.executor_workers)
                logger.info(f"ThreadPoolExecutor initialized with {self.executor_workers} workers.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def update_graph(self, new_data_items: List[Dict[str, Any]]) -> None:
        try:
            await asyncio.gather(*(self._update_single_data(data) for data in new_data_items))
        except Exception as e:
            logger.error(f"Error updating graph: {e}", exc_info=True)
            raise CryptoGMNError("Failed to update graph") from e

    async def _update_single_data(self, new_data: Dict[str, Any]) -> None:
        try:
            validated_data = DataModel(**new_data)
            price = validated_data.c
            volume = validated_data.v
        except ValidationError as e:
            logger.error(f"Data validation error: {e}. Data: {new_data}")
            raise DataValidationError(f"Invalid data format: {e}") from e

        try:
            await asyncio.gather(*(self._update_timeframe(tf, price, volume) for tf in self.timeframes))
        except Exception as e:
            logger.error(f"Error updating timeframes: {e}", exc_info=True)
            raise CryptoGMNError("Failed to update timeframes") from e

    async def _update_timeframe(self, timeframe: str, price: float, volume: float) -> None:
        async with self.locks[timeframe]:
            try:
                data = self.market_data[timeframe]
                data['price'].append(price)
                data['volume'].append(volume)
            except Exception as e:
                logger.error(f"Error updating timeframe {timeframe}: {e}", exc_info=True)
                raise CryptoGMNError(f"Failed to update timeframe {timeframe}") from e

    def get_data(self, timeframe: str, data_type: str) -> Optional[List[Union[float]]]:
        try:
            return list(self.market_data[timeframe][data_type])
        except KeyError:
            logger.warning(f"No data found for timeframe '{timeframe}' and data type '{data_type}'.")
            return None

    def get_all_data(self) -> Dict[str, Dict[str, List[Union[float]]]]:
        return {
            timeframe: {
                data_type: list(data)
                for data_type, data in indicators.items()
            }
            for timeframe, indicators in self.market_data.items()
        }

    async def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        logger.info("CryptoGMN has been shut down gracefully.")
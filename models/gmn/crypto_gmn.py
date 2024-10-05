import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
import logging
from functools import lru_cache
import aiosqlite
import aiohttp
from readerwriterlock import rwlock
import time

from config import CryptoGMNConfig
from logger import logger
from indicators import IndicatorFactory
from data_model import DataModel
from performance_monitor import PerformanceMonitor

class CryptoGMN:
    def __init__(self, config: CryptoGMNConfig):
        self.config = config
        self.timeframes = config.timeframes
        self.indicators = config.indicators
        self.max_history_length = config.max_history
        self.market_data: Dict[str, Dict[str, deque]] = {
            timeframe: {indicator: deque(maxlen=self.max_history_length) for indicator in self.indicators}
            for timeframe in self.timeframes
        }
        self.locks: Dict[str, rwlock.RWLockFair] = {timeframe: rwlock.RWLockFair() for timeframe in self.timeframes}
        self.executor: Optional[ThreadPoolExecutor] = None
        self.indicator_factory = IndicatorFactory()
        self.performance_monitor = PerformanceMonitor()
        self.cache_size = config.cache_size
        self.db_path = config.db_path
        self.queue = asyncio.Queue()

    async def __aenter__(self):
        async with asyncio.Lock():
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=self.config.executor_workers)
                logger.info(f"ThreadPoolExecutor initialized with {self.config.executor_workers} workers.")
        await self.load_persisted_data()
        asyncio.create_task(self.start_periodic_tasks())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def update_graph(self, new_data_items: List[Dict[str, Any]]) -> None:
        start_time = time.time()
        await asyncio.gather(*(self._update_single_data(data) for data in new_data_items))
        end_time = time.time()
        self.performance_monitor.record(start_time, end_time)
        if self.performance_monitor.average_processing_time > self.config.performance_threshold:
            logger.warning(f"Average processing time ({self.performance_monitor.average_processing_time:.4f}s) exceeds threshold.")

    async def _update_single_data(self, new_data: Dict[str, Any]) -> None:
        try:
            validated_data = DataModel(**new_data)
            price = validated_data.c
            volume = validated_data.v
        except ValueError as e:
            logger.error(f"Data validation error: {e}. Data: {new_data}")
            return

        await asyncio.gather(*(self._update_timeframe(tf, price, volume) for tf in self.timeframes))

    async def _update_timeframe(self, timeframe: str, price: float, volume: float) -> None:
        async with self.locks[timeframe].gen_wlock():
            data = self.market_data[timeframe]
            data['price'].append(price)
            data['volume'].append(volume)

            prices_array = np.array(data['price'], dtype=np.float64)

            if len(prices_array) >= 14:
                await self._calculate_indicators(timeframe, prices_array)

    async def _calculate_indicators(self, timeframe: str, prices_array: np.ndarray):
        tasks = []
        for indicator in self.indicators:
            if indicator in ['price', 'volume']:
                continue
            indicator_func = getattr(self.indicator_factory, f"create_{indicator}", None)
            if indicator_func:
                tasks.append(self._calculate_and_append(timeframe, indicator, indicator_func(), prices_array))
        if tasks:
            await asyncio.gather(*tasks)

    async def _calculate_and_append(self, timeframe: str, indicator: str,
                                    indicator_func: Callable[[np.ndarray], Optional[float]],
                                    prices: np.ndarray) -> None:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(self.executor, indicator_func, prices)
            if result is not None:
                self.market_data[timeframe][indicator].append(float(result))
        except Exception as e:
            logger.error(f"Error calculating {indicator} for {timeframe}: {e}")

    @lru_cache(maxsize=100)
    def get_cached_indicator(self, timeframe: str, indicator: str, window: int = 100):
        data = self.get_data(timeframe, indicator)
        if data:
            return tuple(data[-window:])
        return None

    def get_data(self, timeframe: str, indicator: str) -> Optional[List[Union[float, Dict[str, Union[float, bool]]]]]:
        try:
            with self.locks[timeframe].gen_rlock():
                return list(self.market_data[timeframe][indicator])
        except KeyError:
            logger.warning(f"No data found for timeframe '{timeframe}' and indicator '{indicator}'.")
            return None

    def get_all_data(self) -> Dict[str, Dict[str, List[Union[float, Dict[str, Union[float, bool]]]]]]:
        return {
            timeframe: {
                indicator: list(data)
                for indicator, data in indicators.items()
            }
            for timeframe, indicators in self.market_data.items()
        }

    def add_indicator(self, timeframe: str, indicator: str, calculation_func: Callable[[np.ndarray], Optional[Union[float, Dict[str, Union[float, bool]]]]]) -> None:
        if timeframe not in self.market_data:
            logger.error(f"Timeframe '{timeframe}' not found.")
            return
        if indicator in self.market_data[timeframe]:
            logger.warning(f"Indicator '{indicator}' already exists for timeframe '{timeframe}'.")
            return

        self.market_data[timeframe][indicator] = deque(maxlen=self.max_history_length)
        self.indicators.append(indicator)
        logger.info(f"Indicator '{indicator}' added to timeframe '{timeframe}'.")

    def remove_indicator(self, timeframe: str, indicator: str) -> None:
        if timeframe not in self.market_data:
            logger.error(f"Timeframe '{timeframe}' not found.")
            return
        if indicator not in self.market_data[timeframe]:
            logger.warning(f"Indicator '{indicator}' does not exist for timeframe '{timeframe}'.")
            return

        del self.market_data[timeframe][indicator]
        self.indicators.remove(indicator)
        logger.info(f"Indicator '{indicator}' removed from timeframe '{timeframe}'.")

    async def persist_data(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''CREATE TABLE IF NOT EXISTS market_data
                                (timestamp INTEGER, timeframe TEXT, indicator TEXT, value REAL)''')
            for timeframe, indicators in self.market_data.items():
                for indicator, data in indicators.items():
                    await db.executemany('INSERT INTO market_data (timestamp, timeframe, indicator, value) VALUES (?, ?, ?, ?)',
                                         [(int(time.time()), timeframe, indicator, value) for value in data])
            await db.commit()
        logger.info("Market data persisted to the database.")

    async def load_persisted_data(self):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('SELECT timestamp, timeframe, indicator, value FROM market_data ORDER BY timestamp ASC') as cursor:
                async for row in cursor:
                    timestamp, timeframe, indicator, value = row
                    if timeframe in self.market_data and indicator in self.market_data[timeframe]:
                        self.market_data[timeframe][indicator].append(float(value))
        logger.info("Persisted market data loaded from the database.")

    async def fetch_real_time_data(self, exchange: str, symbol: str):
        if exchange.lower() == 'binance':
            url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = float(data['price'])
                        new_data = {'c': price, 'v': 0.0}
                        await self._update_single_data(new_data)
                        logger.info(f"Fetched real-time data from Binance for {symbol}: Price={price}")
                    else:
                        logger.error(f"Failed to fetch data from Binance: {response.status}")

    async def enqueue_data(self, new_data_items: List[Dict[str, Any]]) -> None:
        for data in new_data_items:
            await self.queue.put(data)

    async def batch_processor(self, batch_size: int = 10, timeout: float = 1.0):
        while True:
            batch = []
            try:
                async with asyncio.timeout(timeout):
                    while len(batch) < batch_size:
                        data = await self.queue.get()
                        batch.append(data)
                        self.queue.task_done()
            except asyncio.TimeoutError:
                pass

            if batch:
                try:
                    await self.update_graph(batch)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")

    async def start_periodic_tasks(self):
        asyncio.create_task(self._periodic_cleanup())
        asyncio.create_task(self._periodic_persist())

    async def _periodic_cleanup(self):
        while True:
            # Implement cleanup logic if needed
            await asyncio.sleep(3600)  # Run every hour

    async def _periodic_persist(self):
        while True:
            await self.persist_data()
            await asyncio.sleep(300)  # Persist every 5 minutes

    async def shutdown(self) -> None:
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            logger.info("ThreadPoolExecutor has been shut down gracefully.")
        await self.persist_data()
        logger.info("CryptoGMN has been shut down gracefully.")
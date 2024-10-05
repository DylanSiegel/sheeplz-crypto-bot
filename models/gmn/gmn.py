# File: models/gmn/gmn.py

import networkx as nx
from collections import deque
import numpy as np
import talib
import logging
import asyncio
import concurrent.futures
import threading

class CryptoGMN:
    def __init__(self, timeframes, indicators, max_history_length=1000):
        self.timeframes = timeframes
        self.indicators = indicators
        self.max_history_length = max_history_length
        self.graph = nx.Graph()
        self._initialize_nodes()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.lock = threading.Lock()

    def _initialize_nodes(self):
        for timeframe in self.timeframes:
            for indicator in self.indicators:
                self.graph.add_node(
                    (timeframe, indicator),
                    data=deque(maxlen=self.max_history_length)
                )

    async def update_graph(self, new_data_items):
        """Asynchronously updates the graph with new data."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._update_node_data_batch, new_data_items)

    def _update_node_data_batch(self, new_data_items):
        """Synchronously updates the graph nodes with new data."""
        with self.lock:
            for new_data_item in new_data_items:
                for timeframe in self.timeframes:
                    try:
                        # Update price and volume
                        price = float(new_data_item.get('c', 0.0))  # 'c' for close price
                        volume = float(new_data_item.get('v', 0.0))  # 'v' for volume
                        self.graph.nodes[(timeframe, "price")]['data'].append(price)
                        self.graph.nodes[(timeframe, "volume")]['data'].append(volume)

                        # Recalculate indicators
                        prices = list(self.graph.nodes[(timeframe, "price")]['data'])
                        if len(prices) >= 14:  # Minimum data length for indicators
                            if 'rsi' in self.indicators:
                                rsi = self.calculate_rsi(prices)
                                if rsi is not None:
                                    self.graph.nodes[(timeframe, "rsi")]['data'].append(rsi)
                            if 'macd' in self.indicators:
                                macd = self.calculate_macd(prices)
                                if macd is not None:
                                    self.graph.nodes[(timeframe, "macd")]['data'].append(macd)
                            if 'fibonacci' in self.indicators:
                                fibonacci = self.calculate_fibonacci(prices)
                                if fibonacci is not None:
                                    self.graph.nodes[(timeframe, "fibonacci")]['data'].append(fibonacci)
                    except Exception as e:
                        logging.error(f"Error updating data for timeframe {timeframe}: {e}, Data: {new_data_item}")

    def get_data(self, timeframe, indicator):
        """Retrieves the latest data for a specific timeframe and indicator."""
        with self.lock:
            try:
                data = self.graph.nodes[(timeframe, indicator)]['data']
                return list(data) if data else None
            except KeyError:
                logging.error(f"No data for {timeframe} and {indicator}")
                return None

    def get_all_data(self):
        """Retrieves all market data across all timeframes and indicators."""
        with self.lock:
            market_data = {}
            for timeframe in self.timeframes:
                market_data[timeframe] = {}
                for indicator in self.indicators:
                    data = self.graph.nodes.get((timeframe, indicator), {}).get('data')
                    market_data[timeframe][indicator] = list(data) if data else []
            return market_data

    def shutdown(self):
        """Shuts down the executor gracefully."""
        self.executor.shutdown(wait=True)

    def calculate_rsi(self, prices, period=14):
        """Calculates the Relative Strength Index (RSI)."""
        try:
            rsi = talib.RSI(np.array(prices, dtype=np.float64), timeperiod=period)
            return rsi[-1] if len(rsi) > 0 else None
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            return None

    def calculate_macd(self, prices, fastperiod=12, slowperiod=26, signalperiod=9):
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        try:
            macd, macdsignal, macdhist = talib.MACD(
                np.array(prices, dtype=np.float64),
                fastperiod=fastperiod,
                slowperiod=slowperiod,
                signalperiod=signalperiod
            )
            return macd[-1] if len(macd) > 0 else None
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            return None

    def calculate_fibonacci(self, prices, lookback=14):
        """Calculates the closest Fibonacci retracement level based on the latest price."""
        try:
            if len(prices) < lookback:
                return None
            recent_prices = prices[-lookback:]
            high = max(recent_prices)
            low = min(recent_prices)
            close = prices[-1]
            diff = high - low
            levels = {
                "23.6%": high - 0.236 * diff,
                "38.2%": high - 0.382 * diff,
                "50%": high - 0.5 * diff,
                "61.8%": high - 0.618 * diff,
                "78.6%": high - 0.786 * diff,
                "100%": low
            }
            # Determine which level the close price is closest to
            closest_level = min(levels, key=lambda x: abs(levels[x] - close))
            return levels[closest_level]
        except Exception as e:
            logging.error(f"Error calculating Fibonacci retracement: {e}")
            return None

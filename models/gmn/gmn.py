import networkx as nx
from collections import deque
import numpy as np
# import talib  # We'll replace this
import logging
import asyncio
import concurrent.futures
import threading

# Import the alternative TA library
import pandas_ta as ta

class CryptoGMN:
    """
    Graph Market Network (GMN) for managing and updating cryptocurrency market data.
    Uses networkx to represent market data as a graph.
    Calculates technical indicators using pandas_ta.
    """
    def __init__(self, timeframes, indicators, max_history_length=1000):
        self.timeframes = timeframes
        self.indicators = indicators
        self.max_history_length = max_history_length
        self.graph = nx.Graph()
        self._initialize_nodes()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.lock = threading.Lock()

    def _initialize_nodes(self):
        """Initializes nodes in the graph for each timeframe and indicator."""
        for timeframe in self.timeframes:
            for indicator in self.indicators:
                self.graph.add_node(
                    (timeframe, indicator),
                    data=deque(maxlen=self.max_history_length)
                )

    async def update_graph(self, new_data_items, pool: multiprocessing.Pool):  # Receive the pool
        """Asynchronously updates the graph with new data."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._update_node_data_batch, new_data_items, pool)

    def _update_node_data_batch(self, new_data_items, pool):  # Receive the pool
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

                        # Asynchronous indicator calculations
                        results = []
                        for indicator_name in self.indicators:
                            if indicator_name not in ("price", "volume"):  # Exclude price and volume
                                results.append(pool.apply_async(
                                    self.calculate_indicator,
                                    (list(self.graph.nodes[(timeframe, "price")]['data']), indicator_name)
                                ))

                        for res, indicator_name in zip(results, self.indicators):
                            if indicator_name not in ("price", "volume"):
                                indicator_value = res.get()
                                if indicator_value is not None:
                                    self.graph.nodes[(timeframe, indicator_name)]['data'].append(indicator_value)

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

    def calculate_indicator(self, prices, indicator_name):
        """
        Calculates the specified technical indicator using pandas_ta.
        """
        try:
            if indicator_name == 'rsi':
                return ta.rsi(pd.Series(prices), length=14).iloc[-1]
            elif indicator_name == 'macd':
                macd = ta.macd(pd.Series(prices), fast=12, slow=26, signal=9)
                return macd['MACD_12_26_9'].iloc[-1]  # Access the MACD line
            elif indicator_name == 'fibonacci':
                # pandas_ta doesn't have a direct Fibonacci retracement indicator
                # You'll need to implement your Fibonacci calculation logic here
                return self.calculate_fibonacci(prices)
            else:
                logging.warning(f"Unsupported indicator: {indicator_name}")
                return None
        except Exception as e:
            logging.error(f"Error calculating indicator {indicator_name}: {e}")
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
# File: data/data_processor.py
import asyncio
from typing import Dict, Any, List
import pandas as pd
from .indicator_calculations import IndicatorCalculator
from .storage.data_storage import DataStorage
from error_handler import ErrorHandler
import logging
import aiohttp
import os

class DataProcessor:
    """
    Processes raw kline data, applies technical indicators, and stores the processed data.
    """

    def __init__(
        self,
        data_queue: asyncio.Queue,
        storage: DataStorage,
        indicator_calculator: IndicatorCalculator,
        error_handler: ErrorHandler,
        symbols: List[str],
        timeframes: List[str]
    ):
        """
        Initializes the DataProcessor.

        Args:
            data_queue (asyncio.Queue): Queue to consume raw data from.
            storage (DataStorage): Instance for storing processed data.
            indicator_calculator (IndicatorCalculator): Instance for calculating technical indicators.
            error_handler (ErrorHandler): Instance to handle errors.
            symbols (List[str]): List of trading symbols.
            timeframes (List[str]): List of kline timeframes.
        """
        self.data_queue = data_queue
        self.storage = storage
        self.indicator_calculator = indicator_calculator
        self.error_handler = error_handler
        self.symbols = symbols
        self.timeframes = timeframes
        self.logger = logging.getLogger("DataProcessor")

    async def run(self):
        """
        Continuously consumes data from the queue and processes it.
        """
        while True:
            try:
                data = await self.data_queue.get()
                await self.process_data(data)
                self.data_queue.task_done()
            except asyncio.CancelledError:
                self.logger.info("DataProcessor task cancelled.")
                break
            except Exception as e:
                self.error_handler.handle_error(
                    f"Error in DataProcessor run loop: {e}",
                    exc_info=True,
                    symbol=None,
                    timeframe=None
                )

    async def process_data(self, data: Dict[str, Any]):
        """
        Processes a single batch of kline data.

        Args:
            data (Dict[str, Any]): Raw kline data from the WebSocket.
        """
        try:
            symbol, timeframe = self._extract_symbol_timeframe(data)
            kline_data = self._extract_kline_data(data)
            
            # Load existing data
            existing_df = await self.storage.load_dataframe(symbol, timeframe)
            if existing_df is not None and not existing_df.empty:
                df = pd.concat([existing_df, pd.DataFrame([kline_data])], ignore_index=True)
            else:
                df = pd.DataFrame([kline_data])

            # Remove duplicates based on close_time
            df.drop_duplicates(subset=['close_time'], keep='last', inplace=True)
            # Sort by close_time
            df.sort_values(by='close_time', inplace=True)
            # Reset index
            df.reset_index(drop=True, inplace=True)

            # Calculate indicators
            indicators = self.indicator_calculator.calculate_indicators(symbol, {timeframe: df})

            # Consolidate data
            unified_feed = {
                'symbol': symbol,
                'timeframe': timeframe,
                'data': df.to_dict(orient='records'),
                'indicators': indicators.get(timeframe, {})
            }

            # Store the unified feed
            await self.storage.store_data(unified_feed)
            # Optionally, send to GMN
            await self.send_to_gmn(unified_feed)
            # Live Test Output: Log the unified_feed
            self.logger.info(f"Processed data for {symbol} {timeframe}: {unified_feed}")
        except Exception as e:
            self.error_handler.handle_error(
                f"Error processing data: {e}",
                exc_info=True,
                symbol=None,
                timeframe=None
            )

    def _extract_symbol_timeframe(self, data: Dict[str, Any]) -> tuple[str, str]:
        """
        Extracts the symbol and timeframe from the channel name.

        Args:
            data (Dict[str, Any]): Raw kline data.

        Returns:
            tuple[str, str]: Symbol and timeframe.
        """
        channel = data.get('c', '')
        parts = channel.split('@')
        if len(parts) < 4:
            raise ValueError(f"Invalid channel format: {channel}")
        symbol = parts[2]
        timeframe = parts[3]  # e.g., 'Min30'
        return symbol, timeframe

    def _extract_kline_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts relevant kline data from the message.

        Args:
            data (Dict[str, Any]): Raw kline data.

        Returns:
            Dict[str, Any]: Extracted kline information.
        """
        k = data.get('d', {}).get('k', {})
        if not k:
            raise ValueError("Missing kline data in the message")
        return {
            'open': float(k['o']),
            'high': float(k['h']),
            'low': float(k['l']),
            'close': float(k['c']),
            'volume': float(k['v']),
            'close_time': int(k['T'])
        }

    async def send_to_gmn(self, unified_feed: Dict[str, Any]):
        """
        Sends the unified feed to the GMN module.

        Args:
            unified_feed (Dict[str, Any]): Processed data with indicators.
        """
        gmn_endpoint = os.getenv("GMN_ENDPOINT", "http://localhost:8000/api/gmn")  # Replace with actual endpoint

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(gmn_endpoint, json=unified_feed) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        self.error_handler.handle_error(
                            f"GMN API error: {response.status}, Response: {response_text}",
                            symbol=unified_feed.get('symbol'),
                            timeframe=unified_feed.get('timeframe')
                        )
                    else:
                        self.logger.info(
                            f"Data successfully sent to GMN for {unified_feed.get('symbol')} {unified_feed.get('timeframe')}"
                        )
            except Exception as e:
                self.error_handler.handle_error(
                    f"Error sending data to GMN: {e}",
                    exc_info=True,
                    symbol=unified_feed.get('symbol'),
                    timeframe=unified_feed.get('timeframe')
                )

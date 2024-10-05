# data/processing/data_processor.py
import logging
from typing import Dict, Any, List


class DataProcessor:
    """Processes incoming data and updates the GMN."""

    def __init__(self, gmn):
        """
        Initializes the DataProcessor instance.

        :param gmn: The GMN instance for updating graphs.
        """
        self.gmn = gmn

    async def process_data(self, data_batch: List[Dict[str, Any]]):
        """Processes a batch of data messages."""
        for data in data_batch:
            if 'data' in data and 'channel' in data:
                channel = data['channel']
                if channel.startswith('push.kline'):
                    await self._process_kline_data(data)
                elif channel.startswith("push.private."):
                    await self._process_private_data(data)
            elif 'method' in data:
                self._process_method_data(data)

    async def _process_kline_data(self, data: Dict[str, Any]):
        """Processes kline (candlestick) data."""
        channel_parts = data['channel'].split('.')
        if len(channel_parts) >= 4:
            interval = channel_parts[3]
            kline_data = data['data']

            if kline_data is None:
                logging.warning("Kline data is None. Skipping.")
                return

            if isinstance(kline_data, list):
                for kline in kline_data:
                    kline['interval'] = interval
                await self.gmn.update_graph(kline_data)
            elif isinstance(kline_data, dict):
                kline_data['interval'] = interval
                await self.gmn.update_graph([kline_data])
            else:
                logging.error(f"Unexpected kline data type: {type(kline_data)}. Data: {kline_data}")
        else:
            logging.warning(f"Unexpected channel format: {data['channel']}")

    async def _process_private_data(self, data: Dict[str, Any]):
        """Processes private channel data."""
        channel = data.get("channel")
        if channel == 'push.account':
            logging.info(f"Account Update: {data['data']}")
        elif channel == 'push.order':
            logging.info(f"Order Update: {data['data']}")
        else:
            logging.warning(f"Received data from unhandled private channel: {channel}")

    def _process_method_data(self, data: Dict[str, Any]):
        """Processes messages that contain a method field."""
        method = data['method']
        if method == 'pong':
            logging.debug("Received pong from server.")
        else:
            logging.debug(f"Unhandled message with method: {method}")

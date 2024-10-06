import logging
from typing import Dict, Any, List


class DataProcessor:
    """Processes incoming data and updates the Graph Management Node (GMN)."""

    def __init__(self, gmn):
        self.gmn = gmn

    async def process_data(self, data_batch: List[Dict[str, Any]]):
        """Processes a batch of data messages."""
        for data in data_batch:
            try:
                if 'data' in data and 'channel' in data:
                    channel = data['channel']
                    if channel.startswith('push.kline'):
                        await self._process_kline_data(data)
                    elif channel.startswith("push.private."):
                        await self._process_private_data(data)
                elif 'method' in data:
                    self._process_method_data(data)
            except Exception as e:
                logging.exception(f"Error processing data: {e}")

    async def _process_kline_data(self, data: Dict[str, Any]):
        """Processes kline (candlestick) data."""
        channel_parts = data['channel'].split('.')
        if len(channel_parts) >= 4:
            interval = channel_parts[3]
            kline_data = data['data']

            if not kline_data:
                logging.warning("Kline data is None or empty. Skipping.")
                return

            if isinstance(kline_data, list):  # Batch of klines
                for kline in kline_data:
                    kline['interval'] = interval
                await self.gmn.update_graph(kline_data)
            elif isinstance(kline_data, dict):  # Single kline
                kline_data['interval'] = interval
                await self.gmn.update_graph([kline_data])
            else:
                logging.error(
                    f"Unexpected kline data type: {type(kline_data)}. Data: {kline_data}"
                )
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
            logging.warning(
                f"Unhandled private channel: {channel}. Data: {data.get('data')}"
            )

    def _process_method_data(self, data: Dict[str, Any]):
        """Processes method-based data."""
        method = data.get('method')
        if method == 'pong':
            logging.debug("Received pong from server.")
        elif method and method.startswith('sub'):
            logging.info(f"Subscribed to channel: {method}")
        elif method and method.startswith('unsub'):
            logging.info(f"Unsubscribed from channel: {method}")
        else:
            logging.debug(f"Unhandled method message: {method}")

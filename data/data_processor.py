import asyncio

class DataProcessor:
    def __init__(self, storage, indicator_calculator, error_handler, config):
        self.storage = storage
        self.indicator_calculator = indicator_calculator
        self.error_handler = error_handler
        self.config = config

    async def process_data(self, data_batch):
        """Processes kline data and applies indicators asynchronously."""
        processed_data = {}
        for data in data_batch:
            try:
                kline_data = self._extract_kline_data(data)
                timeframe = self._get_timeframe(data)
                if timeframe not in processed_data:
                    processed_data[timeframe] = []
                processed_data[timeframe].append(kline_data)
            except Exception as e:
                self.error_handler.handle_error(f"Error extracting kline data: {e}", exc_info=True)

        try:
            indicators = await self._calculate_indicators_async(processed_data)
            unified_feed = self._create_unified_feed(processed_data, indicators)
            await self.storage.store_data(unified_feed)
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating indicators or storing data: {e}", exc_info=True)

    async def _calculate_indicators_async(self, processed_data):
        """Calculates indicators asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.indicator_calculator.calculate_indicators, processed_data)

    def _extract_kline_data(self, data):
        # Extract kline data (fill with your logic)
        return {
            'open': data['d']['k']['o'],
            'high': data['d']['k']['h'],
            'low': data['d']['k']['l'],
            'close': data['d']['k']['c'],
            'volume': data['d']['k']['v'],
            'close_time': data['d']['k']['T']
        }

    def _get_timeframe(self, data):
        # Extract timeframe (assuming format is 'spot@public.kline.v3.api@BTCUSDT@kline_1m')
        return data.get('c', '').split('@')[-1].split('_')[-1]

    def _create_unified_feed(self, klines, indicators):
        """Combines kline data and indicators into a unified feed."""
        unified_feed = {}
        for timeframe, data in klines.items():
            unified_feed[timeframe] = {
                'price': [entry['close'] for entry in data],
                'volume': [entry['volume'] for entry in data],
                'open': [entry['open'] for entry in data],
                'high': [entry['high'] for entry in data],
                'low': [entry['low'] for entry in data],
                'close_time': [entry['close_time'] for entry in data],
                'indicators': indicators.get(timeframe, {})
            }
        return unified_feed

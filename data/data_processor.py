import logging
from typing import Dict, Any, List, Union
import pandas as pd
from marshmallow import Schema, fields, validates, ValidationError
from data.config import Config
from data.indicator_calculations import IndicatorCalculator
from data.error_handler import ErrorHandler
from datetime import datetime, timezone


class KlineSchema(Schema):
    """Schema to validate kline data."""
    T = fields.Int(required=True, data_key='T', attribute="close_time")   # Close Time
    a = fields.Float(required=True, data_key='a', attribute="volume")     # Volume
    c = fields.Float(required=True, data_key='c', attribute="close")       # Close Price
    h = fields.Float(required=True, data_key='h', attribute="high")        # High Price
    i = fields.Str(required=True, data_key='i', attribute="timeframe")    # Interval (rename 'i' to 'timeframe')
    l = fields.Float(required=True, data_key='l', attribute="low")         # Low Price
    o = fields.Float(required=True, data_key='o', attribute="open")        # Open Price
    t = fields.Int(required=True, data_key='t', attribute="open_time")    # Open Time
    v = fields.Float(required=True, data_key='v', attribute="quantity")    # Quantity (Base Asset Volume)


class DataProcessor:

    def __init__(self, gmn, indicator_calculator: IndicatorCalculator, error_handler: ErrorHandler, config: Config):
        self.gmn = gmn
        self.indicator_calculator = indicator_calculator
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        self.kline_schema = KlineSchema()  # Initialize the schema once
        self.config = config

    async def process_data(self, data_batch: List[Dict[str, Any]]) -> None:
        """
        Processes a batch of data messages.

        Args:
            data_batch (List[Dict[str, Any]]): List of raw data messages from WebSocket.
        """
        processed_data = {}  # Store kline data organized by timeframe

        for data in data_batch:
            try:
                if data.get('method') == 'PONG':  # Correct ping/pong handling
                    continue  # Heartbeat; ignore and proceed to the next message

                channel = data.get('c')  # Use 'c' for the channel
                if not channel or not channel.startswith("spot@public.kline.v3.api"):
                    continue  # Skip non-kline messages

                # Extract timeframe and other essential fields
                # Assuming the channel format is "spot@public.kline.v3.api@<symbol>@<timeframe>"
                parts = channel.split('@')
                if len(parts) < 4:
                    self.logger.warning(f"Invalid channel format: {channel}")
                    continue

                timeframe = parts[-1].split('.')[0]  # Extract timeframe
                if timeframe not in self.config.timeframes:
                    self.logger.warning(f"Received unsupported timeframe: {timeframe}")
                    continue

                kline_data = data.get('d', {}).get('k', {})  # Accessing the correct path for Kline data
                if not kline_data:  # Skip if Kline data is missing or empty
                    continue

                # Validate the Kline data
                try:
                    validated_kline = self.kline_schema.load(kline_data)

                except ValidationError as e:
                    self.error_handler.handle_error(
                        f"Kline validation error: {e.messages}",
                        data=data,
                        exc_info=True
                    )
                    continue  # Skip this invalid message

                # Organize klines by timeframe
                if timeframe not in processed_data:
                    processed_data[timeframe] = []
                processed_data[timeframe].append(validated_kline)  # Append the validated Kline data

            except Exception as e:
                self.error_handler.handle_error(
                    f"Error processing message: {e}",
                    data=data,
                    exc_info=True
                )

        if not processed_data:
            self.logger.info("No kline data to process in this batch.")
            return  # No kline data in this batch

        try:
            data_frames = {}
            for timeframe, klines in processed_data.items():
                df = pd.DataFrame(klines)
                df = self._add_data_lineage(df, timeframe)
                data_frames[timeframe] = df

            # Calculate indicators
            indicators = self.indicator_calculator.calculate_indicators("BTC_USDT", data_frames)  # Pass data_frames

            # Create unified feed
            unified_feed = self.create_unified_feed(data_frames, indicators)

            # Store unified feed
            await self.gmn.store_data(unified_feed)

        except Exception as e:
            self.error_handler.handle_error(f"Error processing batch data: {e}", exc_info=True)

    def _add_data_lineage(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Adds data lineage and transforms data.

        Args:
            df (pd.DataFrame): DataFrame containing kline data.
            timeframe (str): The timeframe of the kline data.

        Returns:
            pd.DataFrame: Transformed DataFrame with added metadata.
        """
        df['source'] = 'mexc_websocket'
        df['symbol'] = 'BTC_USDT'  # Or get from config if needed
        df['timeframe'] = timeframe
        df['processed_at'] = datetime.now(timezone.utc)
        # Rename columns to more descriptive names if needed
        df = df.rename(columns={
            'c': 'close',
            'v': 'volume',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'T': 'close_time',
            't': 'open_time',
            'a': 'quantity'
        })
        return df

    def create_unified_feed(self, klines: Dict[str, pd.DataFrame], indicators: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combines the kline data and indicators into a single unified feed for GMN.

        Args:
            klines (Dict[str, pd.DataFrame]): Dictionary of DataFrames keyed by timeframe.
            indicators (Dict[str, Dict[str, Any]]): Dictionary of indicators keyed by timeframe.

        Returns:
            Dict[str, Any]: Unified data feed containing all relevant data and indicators.
        """
        unified_feed = {}
        for timeframe, df in klines.items():
            unified_feed[timeframe] = {
                'price': df['close'].tolist(),
                'volume': df['volume'].tolist(),
                'open': df['open'].tolist(),
                'high': df['high'].tolist(),
                'low': df['low'].tolist(),
                'close_time': df['close_time'].tolist(),
                'open_time': df['open_time'].tolist(),
                'quantity': df['quantity'].tolist(),
                'indicators': indicators.get(timeframe, {})
            }
        return unified_feed

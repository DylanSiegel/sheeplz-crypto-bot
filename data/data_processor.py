# File: data/data_processor.py

import logging
from typing import Dict, Any, List
from marshmallow import Schema, fields, validates, ValidationError
from indicator_calculations import IndicatorCalculator
from error_handler import ErrorHandler
from datetime import datetime

class DataSchema(Schema):
    """Schema to validate the incoming data structure."""
    data = fields.Dict(required=True)
    channel = fields.Str(required=True)
    symbol = fields.Str(required=False)  # Optional, default to 'unknown'

    @validates('data')
    def validate_data(self, value):
        if not isinstance(value, dict):
            raise ValidationError('Data must be a dictionary')
        if 'price' not in value or not isinstance(value['price'], (float, int)) or value['price'] <= 0:
            raise ValidationError('Invalid or missing price field in data')
        if 'volume' in value and not isinstance(value['volume'], (float, int)):
            raise ValidationError('Invalid volume field in data')


class DataProcessor:
    def __init__(self, gmn, indicator_calculator: IndicatorCalculator, error_handler: ErrorHandler):
        self.gmn = gmn
        self.indicator_calculator = indicator_calculator
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)

    async def process_data(self, data_batch: List[Dict[str, Any]]) -> None:
        """Processes a batch of data messages."""
        for data in data_batch:
            try:
                # Validate incoming data using the schema
                schema = DataSchema()
                validated_data = schema.load(data)

                if 'kline' in validated_data['channel']:
                    await self._process_kline_data(validated_data)
                else:
                    self.logger.warning(f"Unhandled channel: {validated_data['channel']}")
            except ValidationError as e:
                self.error_handler.handle_error(f"Validation error in data: {e.messages}")
            except Exception as e:
                self.error_handler.handle_error(f"Error processing data: {e}", exc_info=True)

    async def _process_kline_data(self, data: Dict[str, Any]) -> None:
        """Processes kline (candlestick) data."""
        try:
            kline_data = data['data']
            symbol = data.get('symbol', 'unknown')

            # Validate kline_data structure
            if not isinstance(kline_data, dict):
                raise ValueError(f"Invalid kline data format: {kline_data}")

            # Add lineage and transform the data
            transformed_data = self._transform_data(self._add_data_lineage(kline_data))

            # Deduplicate data
            if self._is_duplicate(symbol, '1m', transformed_data):
                self.logger.info(f"Duplicate data for {symbol} at 1m detected, skipping.")
                return

            # Update the graph with the new kline data
            await self.gmn.update_graph([transformed_data])

            # Calculate indicators (ensure enough data is present)
            if self._enough_data_for_indicators(symbol, '1m'):
                indicators = self.indicator_calculator.calculate_indicators(symbol, '1m', transformed_data)
                self.gmn.store_indicators(symbol, '1m', indicators)

        except KeyError as e:
            self.error_handler.handle_error(f"KeyError in processing kline data: {e}")
        except ValueError as e:
            self.error_handler.handle_error(f"ValueError in processing kline data: {e}")
        except Exception as e:
            self.error_handler.handle_error(f"Unhandled error in kline processing: {e}", exc_info=True)

    def _transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms the incoming data if necessary."""
        # Example transformation: round values
        data['price'] = round(data['price'], 2)
        if 'volume' in data:
            data['volume'] = round(data['volume'], 2)
        return data

    def _add_data_lineage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Adds lineage metadata to the data."""
        data['source'] = 'mexc_websocket'
        data['processed_at'] = datetime.utcnow().isoformat()
        return data

    def _is_duplicate(self, symbol: str, timeframe: str, data: Dict[str, Any]) -> bool:
        """Check if the incoming data is a duplicate."""
        existing_data = self.gmn.get_data(symbol, timeframe)
        if existing_data and existing_data[-1]['price'] == data['price'] and existing_data[-1]['volume'] == data.get('volume'):
            return True
        return False

    def _enough_data_for_indicators(self, symbol: str, timeframe: str) -> bool:
        """Checks if there is enough data to calculate indicators."""
        data = self.gmn.get_data(symbol, timeframe)
        if len(data['price']) < 14:  # Example: 14 periods needed for indicators like RSI
            self.logger.info(f"Not enough data to calculate indicators for {symbol} at {timeframe}")
            return False
        return True

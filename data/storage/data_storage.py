import pandas as pd
import os
from typing import Dict, Any


class DataStorage:
    def __init__(self, storage_path: str = "data_storage"):
        """
        Initializes the DataStorage with a specified storage path.

        Args:
            storage_path (str): Directory path where CSV files will be saved.
        """
        self.data = {}
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    def store_data(self, unified_feed: Dict[str, Any]) -> None:
        """
        Stores the unified feed data for all timeframes.

        Args:
            unified_feed (Dict[str, Any]): The unified feed containing data and indicators for each timeframe.
        """
        symbol = "BTC_USDT"  # Assuming single symbol; adjust if multiple symbols

        for timeframe, content in unified_feed.items():
            if symbol not in self.data:
                self.data[symbol] = {}
            if timeframe not in self.data[symbol]:
                self.data[symbol][timeframe] = {}

            # Store price and volume
            self.data[symbol][timeframe]['price'] = content.get('price', [])
            self.data[symbol][timeframe]['volume'] = content.get('volume', [])
            self.data[symbol][timeframe]['open'] = content.get('open', [])
            self.data[symbol][timeframe]['high'] = content.get('high', [])
            self.data[symbol][timeframe]['low'] = content.get('low', [])
            self.data[symbol][timeframe]['close_time'] = content.get('close_time', [])
            self.data[symbol][timeframe]['open_time'] = content.get('open_time', [])
            self.data[symbol][timeframe]['quantity'] = content.get('quantity', [])

            # Store indicators
            self.data[symbol][timeframe]['indicators'] = content.get('indicators', {})

            # Save to file
            self.save_to_file(symbol, timeframe, content)

    def save_to_file(self, symbol: str, timeframe: str, content: Dict[str, Any]) -> None:
        """
        Saves the unified feed to a CSV file per timeframe.

        Args:
            symbol (str): The trading symbol.
            timeframe (str): The timeframe.
            content (Dict[str, Any]): The data and indicators.
        """
        # Prepare a DataFrame
        df = pd.DataFrame({
            'open': content.get('open', []),
            'high': content.get('high', []),
            'low': content.get('low', []),
            'close': content.get('price', []),
            'volume': content.get('volume', []),
            'quantity': content.get('quantity', []),
            'open_time': content.get('open_time', []),
            'close_time': content.get('close_time', []),
        })

        # Add indicators
        indicators = content.get('indicators', {})
        for indicator, values in indicators.items():
            if isinstance(values, dict):
                for sub_indicator, sub_values in values.items():
                    column_name = f"{indicator}_{sub_indicator}"
                    df[column_name] = sub_values
            else:
                df[indicator] = values

        # Define filename
        filename = os.path.join(self.storage_path, f"{symbol}_{timeframe}.csv")
        df.to_csv(filename, index=False)
        self._log_storage(filename)

    def _log_storage(self, filename: str):
        """Logs the storage action."""
        print(f"Data saved to {filename}")

    def get_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Retrieves stored data for a given symbol and timeframe.

        Args:
            symbol (str): The trading symbol.
            timeframe (str): The timeframe.

        Returns:
            pd.DataFrame: DataFrame containing the stored data.
        """
        if symbol in self.data and timeframe in self.data[symbol]:
            data = self.data[symbol][timeframe]
            df = pd.DataFrame({
                'open': data.get('open', []),
                'high': data.get('high', []),
                'low': data.get('low', []),
                'close': data.get('price', []),
                'volume': data.get('volume', []),
                'quantity': data.get('quantity', []),
                'open_time': data.get('open_time', []),
                'close_time': data.get('close_time', []),
            })
            # Add indicators
            indicators = data.get('indicators', {})
            for indicator, values in indicators.items():
                if isinstance(values, dict):
                    for sub_indicator, sub_values in values.items():
                        column_name = f"{indicator}_{sub_indicator}"
                        df[column_name] = sub_values
                else:
                    df[indicator] = values
            return df
        return pd.DataFrame()

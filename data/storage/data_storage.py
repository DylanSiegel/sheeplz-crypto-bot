# data_storage.py
import pandas as pd

class DataStorage:
    def __init__(self):
        self.data = {}

    def store_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        if symbol not in self.data:
            self.data[symbol] = {}
        self.data[symbol][timeframe] = data
        self.save_to_file(symbol, timeframe, data)

    def get_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self.data.get(symbol, {}).get(timeframe)

    def save_to_file(self, symbol: str, timeframe: str, data: pd.DataFrame):
        filename = f"{symbol}_{timeframe}.csv"
        data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
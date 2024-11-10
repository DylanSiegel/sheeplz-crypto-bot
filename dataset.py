import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gc

class MarketDataset(Dataset):
    def __init__(self, data_array, timestamps, sequence_length, price_scaler, indicator_scaler, cache_size=1000):
        self.data_array = data_array
        self.timestamps = timestamps
        self.sequence_length = sequence_length
        self.price_scaler = price_scaler
        self.indicator_scaler = indicator_scaler
        self.max_index = len(data_array) - sequence_length
        self.cache_size = cache_size
        self.cache = {}
        self.n_price_features = len(self.price_scaler.scale_)

    def __len__(self):
        return self.max_index

    def _process_sequence(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        sequence = self.data_array[idx:idx + self.sequence_length]
        price_data = sequence[:, :self.n_price_features]
        indicator_data = sequence[:, self.n_price_features:]

        price_scaled = self.price_scaler.transform(price_data)
        indicator_scaled = self.indicator_scaler.transform(indicator_data)

        combined = np.hstack((price_scaled, indicator_scaled)).astype(np.float32)
        tensor_data = torch.from_numpy(combined)

        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        timestamp = self.timestamps[idx + self.sequence_length - 1]
        timestamp_float = timestamp.timestamp()
        self.cache[idx] = (tensor_data, timestamp_float)

        return tensor_data, timestamp_float

    def __getitem__(self, idx):
        return self._process_sequence(idx)

def load_and_preprocess_data(data_path, scaling_method='standard'):
    df = pd.read_csv(
        data_path,
        parse_dates=['open_time', 'close_time'],
        dtype_backend='pyarrow'
    )

    df['timestamp'] = pd.to_datetime(df['open_time'])
    df.set_index('timestamp', inplace=True)
    timestamps = df.index

    price_columns = ['open', 'high', 'low', 'close', 'volume']
    indicator_columns = [
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
        'macd', 'rsi_14', 'ema_10'
    ]

    analysis_df = df[price_columns + indicator_columns].copy()
    data_array = analysis_df.values

    price_scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
    indicator_scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()

    price_data = data_array[:, :len(price_columns)]
    indicator_data = data_array[:, len(price_columns):]

    price_scaler.fit(np.nan_to_num(price_data))
    indicator_scaler.fit(np.nan_to_num(indicator_data))

    del df
    del analysis_df
    gc.collect()

    return data_array, timestamps, price_columns, indicator_columns, price_scaler, indicator_scaler

# data_provider.py

import os
import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataProvider:
    """
    Production data provider reading from CSV or parquet in chunks.
    """
    def __init__(self, file_path: str, chunk_size: Optional[int] = None):
        self.file_path = file_path
        self.chunk_size = chunk_size
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.endswith((".csv", ".parquet")):
            raise ValueError("File must be CSV or Parquet.")
        self.reader = None
        self._init_reader()
        self.data_iter = None
        self.current_batch = 0

    def _init_reader(self):
        if self.file_path.endswith(".csv"):
            if self.chunk_size is not None:
                self.reader = pd.read_csv(self.file_path, chunksize=self.chunk_size)
            else:
                df = pd.read_csv(self.file_path)
                self.reader = [df]
        else:  # parquet
            df = pd.read_parquet(self.file_path)
            if self.chunk_size is not None:
                self.reader = [df.iloc[i:i+self.chunk_size] for i in range(0, len(df), self.chunk_size)]
            else:
                self.reader = [df]

    def _create_iter(self):
        for df in self.reader:
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
            yield df.values

    def get_next_batch(self) -> Optional[np.ndarray]:
        if self.data_iter is None:
            self.data_iter = self._create_iter()
        try:
            batch = next(self.data_iter)
            self.current_batch += 1
            return batch
        except StopIteration:
            logger.info("No more data from data provider.")
            return None
        except Exception as e:
            logger.error(f"Error reading next batch: {e}")
            return None

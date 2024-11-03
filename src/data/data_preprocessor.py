"""
data_preprocessor.py - Hyper-optimized preprocessor for 1-minute BTC/USDT data
Specifically tuned for Ryzen 9 7900X, 32GB DDR5 RAM, Windows 11
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing as mp
import os
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit, prange, float32, float64
from datetime import datetime
import warnings
import time
import mmap
from concurrent.futures import ThreadPoolExecutor
import threading

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# System-level optimizations for Ryzen 9 7900X
os.environ["OMP_NUM_THREADS"] = "6"  # Half physical cores for optimal NUMA performance
os.environ["NUMBA_NUM_THREADS"] = "6"
pd.set_option('compute.use_numexpr', True)

# CPU affinity optimization for Windows
def set_cpu_affinity():
    """
    Set CPU affinity for optimal NUMA performance.
    Platform-agnostic implementation.
    """
    try:
        # For Windows, Linux, and other platforms
        import psutil
        current_process = psutil.Process()
        
        # Get the number of physical cores (not logical/virtual cores)
        physical_cores = psutil.cpu_count(logical=False)
        
        if physical_cores is None:
            # Fallback if can't determine physical cores
            physical_cores = psutil.cpu_count() // 2
        
        # Calculate the cores to use (first half of physical cores)
        cores_to_use = list(range(physical_cores // 2))
        
        try:
            # Try to set affinity
            current_process.cpu_affinity(cores_to_use)
            logging.info(f"CPU affinity set to cores: {cores_to_use}")
        except Exception as e:
            logging.warning(f"Could not set CPU affinity: {str(e)}")
            
    except Exception as e:
        logging.warning(f"CPU affinity optimization skipped: {str(e)}")

@njit(parallel=True, fastmath=True)
def _handle_outliers_numba(data: np.ndarray, window: int = 24) -> np.ndarray:
    """
    Ultra-optimized outlier handling with SIMD vectorization.
    Specialized for 1-minute BTC price characteristics.
    """
    num_rows, num_cols = data.shape
    cleaned = np.empty_like(data)
    half_window = window // 2

    # Pre-calculate volume bounds for entire array
    volume_bounds = np.zeros(num_cols - 4)  # Only for volume columns
    for i in range(4, num_cols):
        volume_bounds[i-4] = np.quantile(data[:, i], 0.9995)

    # Process each column in parallel
    for col in prange(num_cols):
        is_volume = col >= 4
        upper_bound = volume_bounds[col-4] if is_volume else np.inf
        
        # Vectorized operations for entire column
        if is_volume:
            cleaned[:, col] = np.clip(data[:, col], 0, upper_bound)
        else:
            for row in range(num_rows):
                start_idx = max(0, row - half_window)
                end_idx = min(num_rows, row + half_window + 1)
                window_data = data[start_idx:end_idx, col]
                
                median = np.median(window_data)
                std = np.std(window_data)
                
                low_bound = median - 4 * std
                high_bound = median + 4 * std
                cleaned[row, col] = np.clip(data[row, col], low_bound, high_bound)

    return cleaned

@njit(parallel=True)
def _validate_ohlc_numba(data: np.ndarray) -> np.ndarray:
    """
    Validate OHLC relationships with vectorized operations.
    Specialized for cryptocurrency price characteristics.
    """
    validated = data.copy()
    
    for i in prange(len(data)):
        # Unpack OHLC
        open_, high, low, close = data[i, 0:4]
        
        # Ensure high >= low with vectorized max/min
        if high < low:
            validated[i, 1] = max(high, low)  # High
            validated[i, 2] = min(high, low)  # Low
            
        # Fix open/close prices efficiently
        mid_price = (validated[i, 1] + validated[i, 2]) / 2  # (high + low) / 2
        
        if open_ > validated[i, 1] or open_ < validated[i, 2]:
            validated[i, 0] = mid_price
            
        if close > validated[i, 1] or close < validated[i, 2]:
            validated[i, 3] = mid_price
            
    return validated

class MarketDataPreprocessor:
    """Hyper-optimized preprocessor for 1-minute cryptocurrency data"""
    
    REQUIRED_COLUMNS = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
    ]

    PRICE_COLUMNS = ['open', 'high', 'low', 'close']
    VOLUME_COLUMNS = ['volume', 'quote_asset_volume', 
                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

    def __init__(self, data_path: Union[str, Path], 
                 chunk_size: int = 10_000_000,
                 use_mmap: bool = True):
        """
        Initialize with hardware-optimized settings.
        
        Args:
            data_path: Path to raw data file
            chunk_size: Processing chunk size (adjusted for 32GB RAM)
            use_mmap: Use memory mapping for large files
        """
        self.data_path = Path(data_path)
        self.chunk_size = self._optimize_chunk_size(chunk_size)
        self.use_mmap = use_mmap
        self.data = None
        self.cleaning_stats = {}
        
        # Hardware optimization
        self.num_cores = 6  # Optimal for Ryzen 9 7900X
        self.available_memory = psutil.virtual_memory().available
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_cores)
        
        set_cpu_affinity()
        self._setup_logging()
        
    def _optimize_chunk_size(self, requested_size: int) -> int:
        """Optimize chunk size based on available RAM"""
        available_ram = psutil.virtual_memory().available
        optimal_size = min(requested_size, 
                         int(available_ram * 0.3 / 8))  # 30% of RAM, 8 bytes per float64
        return optimal_size

    def _setup_logging(self) -> None:
        """Setup logging with performance metrics"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('MarketDataPreprocessor')
        self.logger.setLevel(logging.INFO)
        
        handler = RotatingFileHandler(
            log_dir / 'btc_1min_preprocessing.log',
            maxBytes=50*1024*1024,
            backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def load_data(self) -> None:
        """Load data with memory mapping and chunk processing"""
        start_time = time.perf_counter()
        self.logger.info(f"Loading data from {self.data_path}")
        
        try:
            if self.data_path.suffix == '.parquet':
                self._load_parquet()
            else:
                self._load_csv()
                
            # Optimize dtypes for 1-min data
            self._optimize_dtypes()
            
            load_time = time.perf_counter() - start_time
            self.logger.info(f"Loaded {len(self.data):,} rows in {load_time:.2f}s")
            self._log_memory_usage()
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _load_parquet(self) -> None:
        """Optimized Parquet loading"""
        table = pq.read_table(
            self.data_path,
            memory_map=self.use_mmap,
            use_threads=True
        )
        self.data = table.to_pandas(
            split_blocks=True,
            self_destruct=True,
            use_threads=True
        )

    def _load_csv(self) -> None:
        """Optimized CSV loading with chunking"""
        chunks = []
        total_rows = 0
        
        # Calculate optimal chunk size
        chunk_size = self._optimize_chunk_size(self.chunk_size)
        
        for chunk in pd.read_csv(
            self.data_path,
            chunksize=chunk_size,
            usecols=self.REQUIRED_COLUMNS,
            dtype={col: 'float32' for col in self.PRICE_COLUMNS + self.VOLUME_COLUMNS},
            parse_dates=['open_time']
        ):
            chunks.append(chunk)
            total_rows += len(chunk)
            self.logger.info(f"Loaded chunk: {total_rows:,} rows")
            
        self.data = pd.concat(chunks, ignore_index=True)

    def _optimize_dtypes(self) -> None:
        """Optimize DataFrame dtypes for memory efficiency"""
        for col in self.PRICE_COLUMNS + self.VOLUME_COLUMNS:
            self.data[col] = self.data[col].astype('float32')
        self.data['number_of_trades'] = self.data['number_of_trades'].astype('uint16')
        
        self.data.set_index('open_time', inplace=True)
        self.data.sort_index(inplace=True)

    def clean_data(self) -> pd.DataFrame:
        """Clean data using parallel processing"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        start_time = time.perf_counter()
        self.logger.info("Starting parallel data cleaning")
        
        try:
            # Split data for parallel processing
            chunks = np.array_split(self.data, self.num_cores)
            
            # Process chunks in parallel
            with mp.Pool(processes=self.num_cores) as pool:
                cleaned_chunks = pool.map(self._clean_chunk, chunks)
            
            # Combine results
            self.data = pd.concat(cleaned_chunks)
            self.data.sort_index(inplace=True)
            
            clean_time = time.perf_counter() - start_time
            self.logger.info(f"Cleaning completed in {clean_time:.2f}s")
            self._log_memory_usage()
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise

    def _clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Clean a data chunk with Numba acceleration"""
        # Convert to numpy array for Numba processing
        data = chunk[self.PRICE_COLUMNS + self.VOLUME_COLUMNS].values
        
        # Clean outliers
        cleaned_data = _handle_outliers_numba(data)
        
        # Validate OHLC
        cleaned_data[:, 0:4] = _validate_ohlc_numba(cleaned_data[:, 0:4])
        
        # Update DataFrame
        chunk[self.PRICE_COLUMNS + self.VOLUME_COLUMNS] = cleaned_data
        return chunk

    def save_processed_data(self, output_path: Union[str, Path]) -> None:
        """Save data with optimal partitioning"""
        if self.data is None:
            raise ValueError("No processed data available")
            
        start_time = time.perf_counter()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Partition by year and month for efficient querying
        self.data.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            row_group_size=100_000,
            partition_cols=['year', 'month']
        )
        
        save_time = time.perf_counter() - start_time
        self.logger.info(f"Saved data in {save_time:.2f}s")

    def _log_memory_usage(self) -> None:
        """Log detailed memory usage"""
        if self.data is not None:
            usage = self.data.memory_usage(deep=True)
            total = usage.sum() / 1024**2
            self.logger.info(f"Memory usage: {total:.2f} MB")
            self.logger.info(f"Available RAM: {psutil.virtual_memory().available/1024**3:.1f} GB")
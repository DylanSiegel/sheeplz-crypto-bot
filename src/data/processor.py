import modin.pandas as mpd
import cupy as cp
import dask.array as da
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import RobustScaler
import ray
from loguru import logger
from numba import jit, cuda
import psutil
from functools import partial
from concurrent.futures import ThreadPoolExecutor

@ray.remote
class DistributedFeatureCalculator:
    """Distributed feature calculation using Ray"""
    def __init__(self, config):
        self.config = config
        
    def calculate_features(self, chunk):
        return self.calculate_all_features(chunk)

class EnhancedDataProcessor:
    """Enhanced data processor with distributed computing and GPU acceleration"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Ray for distributed processing
        if not ray.is_initialized():
            ray.init(num_cpus=psutil.cpu_count())
        
        # Configure logging
        logger.add(
            f"logs/data_processor_{self.config.timeframes[0]}.log",
            rotation="500 MB"
        )
        
        # Initialize caches and buffers
        self._data_cache = {}
        self._feature_cache = {}
        self._scaler_cache = {}
        
        # Initialize GPU memory pool if available
        if torch.cuda.is_available():
            self.gpu_memory_pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(self.gpu_memory_pool.malloc)
    
    def _load_timeframe_optimized(self, timeframe: str) -> mpd.DataFrame:
        """Load timeframe data using Modin for parallel processing"""
        if timeframe not in self._data_cache:
            file_path = self.config.raw_dir / f"btc_{timeframe}_data_2018_to_2024-2024-10-10.csv"
            
            # Define dtypes for efficient memory usage
            dtypes = {
                'Open': np.float32,
                'High': np.float32,
                'Low': np.float32,
                'Close': np.float32,
                'Volume': np.float32,
                'Quote asset volume': np.float32,
                'Number of trades': np.int32,
                'Taker buy base asset volume': np.float32,
                'Taker buy quote asset volume': np.float32
            }
            
            # Use Modin's parallel read_csv
            df = mpd.read_csv(
                file_path,
                parse_dates=['Open time'],
                dtype=dtypes,
                engine='pyarrow'  # Use PyArrow engine for better performance
            )
            
            df.set_index('Open time', inplace=True)
            self._data_cache[timeframe] = df.sort_index()
            
            logger.info(f"Loaded {timeframe} data: {len(df)} rows")
            
        return self._data_cache[timeframe]
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _prepare_sequences_numba(feature_array: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences using Numba acceleration"""
        n_sequences = len(feature_array) - sequence_length
        X = np.empty((n_sequences, sequence_length, feature_array.shape[1]), dtype=np.float32)
        y = np.empty(n_sequences, dtype=np.float32)
        
        for i in range(n_sequences):
            X[i] = feature_array[i:i + sequence_length]
            y[i] = feature_array[i + sequence_length, 0]
            
        return X, y
    
    def _normalize_features_gpu(
        self,
        features: Dict[str, np.ndarray],
        scaler: Optional[RobustScaler] = None,
        fit: bool = True
    ) -> Tuple[Dict[str, np.ndarray], RobustScaler]:
        """Normalize features using GPU acceleration"""
        # Convert features to CuPy array
        feature_array = cp.array(np.column_stack(list(features.values())), dtype=cp.float32)
        
        if scaler is None and fit:
            scaler = RobustScaler()
            normalized = cp.array(scaler.fit_transform(feature_array.get()))
        else:
            normalized = cp.array(scaler.transform(feature_array.get()))
        
        # Convert back to dictionary
        normalized_features = {}
        for i, key in enumerate(features.keys()):
            normalized_features[key] = normalized[:, i].get()
            
        return normalized_features, scaler
    
    def process_timeframe_distributed(
        self,
        timeframe: str,
        save: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process timeframe data using distributed computing"""
        logger.info(f"Processing {timeframe} timeframe...")
        
        # Load data using Modin
        df = self._load_timeframe_optimized(timeframe)
        
        # Distribute feature calculation using Ray
        num_partitions = psutil.cpu_count()
        df_splits = np.array_split(df, num_partitions)
        
        # Create remote feature calculators
        calculators = [DistributedFeatureCalculator.remote(self.config) for _ in range(num_partitions)]
        
        # Calculate features in parallel
        feature_futures = [calculator.calculate_features.remote(split) for calculator, split in zip(calculators, df_splits)]
        feature_results = ray.get(feature_futures)
        
        # Merge feature results
        features = {}
        for key in feature_results[0].keys():
            features[key] = np.concatenate([result[key] for result in feature_results])
        
        # Normalize features using GPU
        if timeframe not in self._scaler_cache:
            features, scaler = self._normalize_features_gpu(features, fit=True)
            self._scaler_cache[timeframe] = scaler
        else:
            features, _ = self._normalize_features_gpu(features, self._scaler_cache[timeframe], fit=False)
        
        # Prepare sequences using Numba
        X, y = self._prepare_sequences_numba(
            np.column_stack(list(features.values())),
            self.config.sequence_length
        )
        
        # Convert to torch tensors with optimal memory format
        X = torch.from_numpy(X).to(
            device=self.device,
            memory_format=torch.channels_last
        )
        y = torch.from_numpy(y).to(self.device)
        
        # Split into train and validation
        split_idx = int(len(X) * self.config.train_ratio)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        if save:
            self._save_processed_data(timeframe, (X_train, y_train, X_val, y_val))
        
        return X_train, y_train, X_val, y_val
    
    def create_optimized_dataloaders(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create optimized dataloaders with prefetching and pinned memory"""
        
        class OptimizedDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y
                
            def __len__(self):
                return len(self.X)
                
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        # Calculate optimal batch size based on available memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        system_mem = psutil.virtual_memory().total
        
        optimal_batch_size = min(
            self.config.batch_size,
            int(min(gpu_mem, system_mem) * 0.1 / (X_train[0].numel() * X_train[0].element_size()))
        )
        
        logger.info(f"Using optimal batch size: {optimal_batch_size}")
        
        train_dataset = OptimizedDataset(X_train, y_train)
        val_dataset = OptimizedDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=optimal_batch_size,
            shuffle=True,
            num_workers=psutil.cpu_count() // 2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            generator=torch.Generator(device=self.device)
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=optimal_batch_size * 2,
            shuffle=False,
            num_workers=psutil.cpu_count() // 4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        return train_loader, val_loader
    
    def cleanup(self):
        """Clean up resources"""
        # Clear caches
        self._data_cache.clear()
        self._feature_cache.clear()
        self._scaler_cache.clear()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.gpu_memory_pool.free_all_blocks()
        
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            
        logger.info("Cleaned up all resources")
import pandas as pd
import numpy as np
import os
from ta import trend, momentum, volatility, volume
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.utils.data
from numba import jit, cuda
import cupy as cp
from functools import partial, lru_cache
from pathlib import Path
import psutil
import warnings
from tqdm.auto import tqdm
import gc
from typing import Dict, List, Tuple
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OptimizedGPUPreprocessor:
    def __init__(self):
        self.num_cpu_cores = max(1, (psutil.cpu_count(logical=False) or 2) - 1)  # Leave one core free
        self.num_gpu_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.batch_size = self._calculate_optimal_batch_size()
        self.raw_data_paths = {
            "15m": Path("data/raw/btc_15m_data_2018_to_2024-2024-10-10.csv"),
            "1h": Path("data/raw/btc_1h_data_2018_to_2024-2024-10-10.csv"),
            "4h": Path("data/raw/btc_4h_data_2018_to_2024-2024-10-10.csv"),
            "1d": Path("data/raw/btc_1d_data_2018_to_2024-2024-10-10.csv")
        }
        self.processed_data_dir = Path("data/final")
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self._init_cuda_kernels()
        
    @property
    def columns_to_use(self) -> List[str]:
        return [
            'Open time',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'Close time',
            'Quote asset volume',
            'Number of trades',
            'Taker buy base asset volume',
            'Taker buy quote asset volume'
        ]
        
    @property
    def column_dtypes(self) -> Dict[str, np.dtype]:
        return {
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

    def _init_cuda_kernels(self):
        if self.num_gpu_devices > 0:
            try:
                self.device = cp.cuda.Device(0)
                self.stream = cp.cuda.Stream()
            except Exception as e:
                logging.warning(f"Failed to initialize CUDA: {str(e)}")
                self.num_gpu_devices = 0

    @staticmethod
    def _calculate_optimal_batch_size() -> int:
        try:
            total_ram = psutil.virtual_memory().total
            gpu_memory = (torch.cuda.get_device_properties(0).total_memory 
                         if torch.cuda.is_available() else 0)
            available_memory = max(total_ram, gpu_memory)
            # More conservative batch size
            return min(int(available_memory * 0.05 / (1024 * 1024)), 100000)
        except:
            return 50000  # Safe default

    def _parallel_process_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            with ThreadPoolExecutor(max_workers=self.num_cpu_cores) as executor:
                futures = []
                
                # Calculate basic indicators
                futures.append(executor.submit(self._calculate_price_indicators, df.copy()))
                futures.append(executor.submit(self._calculate_volume_indicators, df.copy()))
                futures.append(executor.submit(self._calculate_momentum_indicators, df.copy()))
                
                results = []
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logging.error(f"Error in parallel processing: {str(e)}")
                        continue
                
                # Merge results
                for result in results:
                    df = df.join(result)
                
            return df
        except Exception as e:
            logging.error(f"Error in parallel processing: {str(e)}")
            return df

    def process_file(self, timeframe: str):
        try:
            logging.info(f"Starting processing for {timeframe}")
            
            if not self.raw_data_paths[timeframe].exists():
                logging.error(f"Input file not found: {self.raw_data_paths[timeframe]}")
                return
            
            # Process in smaller chunks
            chunks = []
            total_rows = sum(1 for _ in open(self.raw_data_paths[timeframe])) - 1
            chunk_size = min(self.batch_size, total_rows // 10)  # Ensure at least 10 chunks
            
            for df_chunk in tqdm(
                pd.read_csv(
                    self.raw_data_paths[timeframe],
                    usecols=self.columns_to_use,
                    dtype=self.column_dtypes,
                    parse_dates=['Open time', 'Close time'],
                    chunksize=chunk_size
                ),
                desc=f"Processing {timeframe}",
                total=(total_rows // chunk_size) + 1
            ):
                try:
                    # Convert column names to lowercase
                    df_chunk.columns = df_chunk.columns.str.lower()
                    
                    # Process chunk
                    if self.num_gpu_devices > 0:
                        processed_chunk = self._preprocess_chunk_gpu(df_chunk, timeframe)
                    else:
                        processed_chunk = self._preprocess_chunk_cpu(df_chunk, timeframe)
                    
                    chunks.append(processed_chunk)
                    
                    # Explicit cleanup
                    del df_chunk
                    del processed_chunk
                    gc.collect()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logging.error(f"Error processing chunk in {timeframe}: {str(e)}")
                    continue
            
            if not chunks:
                logging.error(f"No chunks were successfully processed for {timeframe}")
                return
            
            # Combine chunks and save
            logging.info(f"Combining chunks for {timeframe}")
            df_processed = pd.concat(chunks, ignore_index=True)
            self._add_target_variables(df_processed, timeframe)
            
            # Save to compressed parquet
            output_path = self.processed_data_dir / f"processed_data_{timeframe}.parquet"
            df_processed.to_parquet(output_path, compression='snappy')
            
            logging.info(f"Successfully processed {timeframe}")
            
            # Final cleanup
            del df_processed
            del chunks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Error processing file {timeframe}: {str(e)}")
            raise

    def process_all_files(self):
        """Process all timeframe files sequentially to avoid memory issues"""
        for timeframe in self.raw_data_paths.keys():
            try:
                self.process_file(timeframe)
            except Exception as e:
                logging.error(f"Failed to process {timeframe}: {str(e)}")
                continue

if __name__ == "__main__":
    try:
        warnings.filterwarnings('ignore')
        preprocessor = OptimizedGPUPreprocessor()
        preprocessor.process_all_files()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        sys.exit(1)
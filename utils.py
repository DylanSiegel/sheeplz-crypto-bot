# utils.py

import cudf
import dask_cudf
from concurrent.futures import ThreadPoolExecutor
from rolling_utils import calculate_rolling_statistics
from feature_engineering import calculate_indicators
import pandas as pd
import logging
import gc

def optimize_dataframe(df: cudf.DataFrame) -> cudf.DataFrame:
    """
    Optimizes DataFrame memory usage by downcasting numerical columns and converting object types to categorical.
    
    Args:
        df (cudf.DataFrame): Input DataFrame to optimize.
    
    Returns:
        cudf.DataFrame: Optimized DataFrame with reduced memory footprint.
    """
    try:
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')  # Downcast float64 to float32
            elif df[col].dtype == 'int64':
                df[col].astype('int32')  # Downcast int64 to int32
            elif df[col].dtype == 'object':
                df[col] = df[col].astype('category')  # Convert object types to categorical
        gc.collect()  # Force garbage collection to free memory
        return df
    except Exception as e:
        logging.error(f"Error in optimize_dataframe: {e}", exc_info=True)
        raise

def process_chunk(filepath_chunk: list) -> cudf.DataFrame:
    """
    Processes a list of filepaths by reading, merging, and calculating technical indicators.
    
    Args:
        filepath_chunk (list): List of file paths to process.
    
    Returns:
        cudf.DataFrame: Processed DataFrame with technical indicators.
    
    Raises:
        ValueError: If no valid data frames are processed from the chunk.
    """
    try:
        # Read CSV files using Dask-CuDF, handling missing values
        ddf = dask_cudf.read_csv(
            filepath_chunk, 
            dtype={'Volume': 'float64', 'Trades': 'float64'}, 
            assume_missing=True
        )
        # Compute the Dask DataFrame and sort by 'Open time'
        merged_df_chunk = ddf.compute().sort_values(by='Open time')
        
        if merged_df_chunk.empty:
            raise ValueError("No valid data frames processed from chunk.")
        
        # Calculate technical indicators
        processed = calculate_indicators(merged_df_chunk)
        
        # Free memory by deleting intermediate objects
        del ddf, merged_df_chunk
        gc.collect()
        
        return processed
    except Exception as e:
        # Log the specific error and the filepaths that caused it
        logging.error(f"Error processing files {filepath_chunk}: {e}", exc_info=True)
        raise  # Reraise exception for visibility in the main loop

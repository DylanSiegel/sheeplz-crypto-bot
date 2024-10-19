# main.py
import time
import logging
from logging_config import setup_logging
import glob
import os
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
from data_transformation import calculate_fft, perform_pca, handle_missing_values
from feature_engineering import calculate_indicators, calculate_wavelet_features

# Set up logging configuration
setup_logging()

def process_chunk(chunk):
    """
    Process a chunk of data.
    
    :param chunk: List of file paths
    :return: Processed pandas DataFrame
    """
    try:
        # Efficient Dask DataFrame Concatenation
        df = pd.concat(dd.read_csv(chunk).to_delayed(), ignore_index=True)
        
        # Calculate technical indicators
        df = calculate_indicators(df)
        
        # Perform FFT
        fft_features = calculate_fft(df['Close'])
        df = df.join(fft_features)
        
        # Perform PCA
        pca_result = perform_pca(df, n_components=5)
        df = df.join(pca_result)
        
        # Calculate Wavelet Features
        close_numpy = df['Close'].values
        close_wavelet_features = calculate_wavelet_features(close_numpy)
        
        if close_wavelet_features.size > 0:
            wavelet_columns = [f'wavelet_{i}' for i in range(close_wavelet_features.size)]
            df = pd.concat([df, pd.DataFrame(close_wavelet_features.reshape(1, -1), columns=wavelet_columns)], axis=1)
        
        return df  # Return the processed chunk as a Pandas DataFrame
    
    except FileNotFoundError:
        logging.error(f"Error: One or more files in the chunk not found: {chunk}")
        return pd.DataFrame()  # Return empty DataFrame in case of error
    
    except Exception as e:
        logging.error(f"Error processing chunk {chunk}: {e}")
        return pd.DataFrame()


def validate_dataframe(df):
    """
    Validate the processed DataFrame.
    
    :param df: Input pandas DataFrame
    :return: Boolean indicating validation success
    """
    required_columns = ['Close', 'Volume', 'RSI', 'MACD']
    for col in required_columns:
        if col not in df.columns or df[col].isnull().any():
            logging.error(f"Validation failed: Missing or NaN values in column {col}")
            return False
    return True


def main():
    setup_logging()
    start_time = time.time()

    try:
        # Set up Dask cluster with efficient configuration
        cluster = LocalCluster(n_workers=min(12, os.cpu_count()),  # Adjust based on available CPU cores
                               threads_per_worker=2,  # Balance between parallelism and overhead
                               processes=True,  # Use separate processes for true parallelism
                               memory_limit='16GB')  # Ensure sufficient memory for workers
        client = Client(cluster)  # Connect to the cluster
        logging.info("Dask cluster initialized.")

        # Define data paths
        RAW_DATA_PATH = r"C:\Users\dylan\Desktop\DATA-LAKE\data\raw\*.csv"
        PROCESSED_DATA_PATH = r"C:\Users\dylan\Desktop\DATA-LAKE\data\processed\btc_merged_advanced_features.parquet"

        # Gather file paths in chunks
        file_paths = glob.glob(RAW_DATA_PATH)
        chunk_size = 1000  # Adjust based on system memory and processing capacity
        for i in range(0, len(file_paths), chunk_size):
            chunk = file_paths[i:i + chunk_size]

            # Process and write data in chunks, utilizing Dask's parallel processing
            futures = client.map(process_chunk, chunk)
            dfs = client.gather(futures)
            dfs = [df for df in dfs if not df.empty]  # Filter out empty DataFrames
            
            if len(dfs) > 0:
                combined = pd.concat(dfs, ignore_index=True)
                
                # Validate the combined DataFrame
                if validate_dataframe(combined):
                    # Write to parquet, appending if the file exists
                    combined.to_parquet(PROCESSED_DATA_PATH, engine='pyarrow', compression='snappy', append=os.path.exists(PROCESSED_DATA_PATH))
                    logging.info(f"Processed and written chunk {i // chunk_size + 1} of {len(file_paths) // chunk_size + 1}")
                else:
                    logging.error(f"Validation failed for chunk {i // chunk_size + 1}")

    except Exception as e:
        logging.error(f"Error in main processing loop: {e}", exc_info=True)
    finally:
        try:
            client.close()
            cluster.close()  # Close the cluster as well.
            logging.info("Dask cluster and client closed.")
        except:
            logging.warning("Dask client/cluster was not initialized or already closed.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
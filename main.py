# main.py

import time  # Import time

def main():
    setup_logging()
    start_time = time.time()  # Record start time
    
    try:
        # Initialize Dask cluster with GPU support
        cluster = LocalCluster(
            n_workers=min(4, os.cpu_count()),  # Adjust based on GPU resources
            threads_per_worker=1,
            processes=True,
            memory_limit='28GB'  # Adjust based on system
        )
        client = Client(cluster)
        
        logging.info("Dask cluster initialized.")
        
        # Define data paths
        RAW_DATA_PATH = r"C:\Users\dylan\Desktop\DATA-LAKE\data\raw\*.csv"
        PROCESSED_DATA_PATH = r"C:\Users\dylan\Desktop\DATA-LAKE\data\processed\btc_merged_advanced_features.parquet"
        
        # Get list of file paths
        file_paths = glob.glob(RAW_DATA_PATH)
        output_path = PROCESSED_DATA_PATH
        
        chunk_size = 10
        num_chunks = (len(file_paths) + chunk_size - 1) // chunk_size
        batch_size = 50  # Number of chunks to process before writing
        
        batch_chunks = []
        
        for i in tqdm(range(num_chunks), desc="Processing Chunks"):
            chunk = file_paths[i * chunk_size:(i + 1) * chunk_size]
            try:
                # Submit the processing task to Dask
                future = client.submit(process_chunk, chunk)
                processed_chunk = future.result()
                
                # Optimize the processed DataFrame
                processed_chunk = optimize_dataframe(processed_chunk)
                
                # Validate the processed DataFrame
                if validate_dataframe(processed_chunk):
                    batch_chunks.append(processed_chunk)
                else:
                    logging.error(f"Validation failed for chunk {i}.")
                
                # Write batch to Parquet if batch_size is reached
                if len(batch_chunks) >= batch_size:
                    combined = cudf.concat(batch_chunks, ignore_index=True)
                    combined.to_parquet(
                        output_path,
                        engine='pyarrow',
                        compression='snappy',
                        append=os.path.exists(output_path)
                    )
                    logging.info(f"Written batch {i//batch_size} to Parquet.")
                    batch_chunks = []
            except Exception as e:
                logging.error(f"Error processing chunk {i}: {e}", exc_info=True)
        
        # Write any remaining chunks after loop
        if batch_chunks:
            combined = cudf.concat(batch_chunks, ignore_index=True)
            combined.to_parquet(
                output_path,
                engine='pyarrow',
                compression='snappy',
                append=os.path.exists(output_path)
            )
            logging.info("Final batch written to Parquet.")
    
    except Exception as e:
        logging.error(f"Error in main processing loop: {e}", exc_info=True)
    finally:
        # Ensure Dask client is closed
        try:
            client.close()
            logging.info("Dask cluster closed.")
        except:
            logging.warning("Dask client was not initialized or already closed.")
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    logging.info(f"Total processing time: {elapsed_time:.2f} seconds")  # Log the execution time

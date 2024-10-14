# utils.py
import cudf
from concurrent.futures import ThreadPoolExecutor
from feature_engineering import calculate_indicators, calculate_rolling_statistics


def optimize_dataframe(df):
    for col in df.select_dtypes(include=['float64']).columns:
        if df[col].memory_usage() > df[col].astype('float32').memory_usage():
            df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].memory_usage() > df[col].astype('int32').memory_usage():
            df[col] = df[col].astype('int32')
    return df


def process_chunk(filepath_chunk, n_jobs):
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        df_list_chunk = []
        for future in executor.map(cudf.read_csv, filepath_chunk):
            try:
                df_list_chunk.append(future)
            except Exception as e:
                print(f"Error processing file: {e}")
    
    if df_list_chunk:
        merged_df_chunk = cudf.concat(df_list_chunk, ignore_index=True).sort_values(by='Open time')
        del df_list_chunk  # Free memory immediately after concatenation
        return calculate_indicators(merged_df_chunk)
    else:
        raise ValueError("No valid data frames processed from chunk.")
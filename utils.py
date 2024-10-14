# utils.py
from imports import *

def optimize_dataframe(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def process_chunk(filepath_chunk, n_jobs):
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        df_list_chunk = list(executor.map(cudf.read_csv, filepath_chunk))
    merged_df_chunk = cudf.concat(df_list_chunk, ignore_index=True).sort_values(by='Open time')
    return engineer_features(merged_df_chunk)

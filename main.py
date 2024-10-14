// File: main.py
# main.py
import os
import glob
from utils import optimize_dataframe, process_chunk
from tqdm import tqdm


def main():
    N_JOBS = min(24, os.cpu_count())
    file_paths = glob.glob(r"C:\Users\dylan\Desktop\DATA-LAKE\data\raw\*.csv")
    output_path = r"C:\Users\dylan\Desktop\DATA-LAKE\data\processed\btc_merged_advanced_features.csv"

    chunk_size = 10
    num_chunks = (len(file_paths) + chunk_size - 1) // chunk_size

    for i in tqdm(range(num_chunks), desc="Processing Chunks"):
        chunk = file_paths[i * chunk_size:(i + 1) * chunk_size]
        processed_chunk = process_chunk(chunk, N_JOBS)
        processed_chunk = optimize_dataframe(processed_chunk)
        processed_chunk.to_csv(
            output_path,
            mode="a",
            header=not os.path.exists(output_path),
            index=False,
        )


if __name__ == "__main__":
    main()

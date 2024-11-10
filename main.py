import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import logging
from tqdm.auto import tqdm
import multiprocessing as mp
import os
import sys

from ai_module import HypersphericalEncoder
from dataset import MarketDataset, load_and_preprocess_data
from visualization import Visualizer

def setup_logging():
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler("market_visualizer_optimized.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

logger = setup_logging()

def get_num_workers():
    if os.name == 'nt':
        return min(4, mp.cpu_count())
    return min(mp.cpu_count(), 8)

NUM_WORKERS = get_num_workers()
logger.info(f"Using {NUM_WORKERS} workers for data loading")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def main():
    try:
        data_path = "data/raw/btc_usdt_1m_processed.csv"
        batch_size = 512
        sequence_length = 64
        hidden_size = 256
        scaling_method = "minmax"
        cache_size = 1000
        use_amp = True

        data_array, timestamps, price_columns, indicator_columns, price_scaler, indicator_scaler = load_and_preprocess_data(data_path, scaling_method)

        dataset = MarketDataset(
            data_array=data_array,
            timestamps=timestamps,
            sequence_length=sequence_length,
            price_scaler=price_scaler,
            indicator_scaler=indicator_scaler,
            cache_size=cache_size
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if NUM_WORKERS > 0 else False
        )

        encoder = HypersphericalEncoder(
            projection_dim=hidden_size,
            sequence_length=sequence_length,
            n_price_features=len(price_columns),
            n_indicator_features=len(indicator_columns),
            device=device
        )

        available_sequences = len(dataset)
        n_samples = 10000
        max_samples = min(n_samples, available_sequences)

        np.random.seed(42)
        indices = np.random.choice(dataset.max_index, size=max_samples, replace=False)

        subset_dataset = Subset(dataset, indices)
        subset_dataloader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        encoded_states = []
        timestamps_list = []

        for batch_sequences, batch_timestamps in tqdm(subset_dataloader, desc="Encoding sequences"):
            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    batch_encoded, _ = encoder.encode_batch(batch_sequences)
            else:
                batch_encoded, _ = encoder.encode_batch(batch_sequences)

            encoded_states.append(batch_encoded.cpu().numpy())
            timestamps_list.extend(batch_timestamps.tolist())

        all_encoded = np.vstack(encoded_states)

        save_encoded = True
        encoded_save_path = "data/encoded_hypersphere.npy"
        if save_encoded:
            save_path = Path(encoded_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, all_encoded)
            logger.info(f"Encoded data saved to {save_path}")

        visualizer = Visualizer(
            data_array=data_array,
            timestamps=timestamps,
            price_columns=price_columns,
            indicator_columns=indicator_columns,
            sequence_length=sequence_length
        )

        output_dir = Path("visualizations")
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory: {output_dir.resolve()}")

        with tqdm(total=2, desc="Generating visualizations") as pbar:
            fig_tsne = visualizer.visualize(
                encoded_states=all_encoded,
                timestamps=timestamps_list,
                indices=indices,
                color_feature="close",
                hover_features=['open', 'high', 'low', 'volume'],
                projection_method='tsne',
                create_subplots=True
            )

            if fig_tsne:
                output_path_tsne = output_dir / "market_visualization_tsne.html"
                fig_tsne.write_html(output_path_tsne)
                logger.info(f"t-SNE visualization saved at {output_path_tsne.resolve()}")
            pbar.update(1)

            fig_pca = visualizer.visualize(
                encoded_states=all_encoded,
                timestamps=timestamps_list,
                indices=indices,
                color_feature='rsi_14',
                hover_features=price_columns + indicator_columns,
                projection_method='pca',
                create_subplots=True
            )

            if fig_pca:
                output_path_pca = output_dir / "market_visualization_pca.html"
                fig_pca.write_html(output_path_pca)
                logger.info(f"PCA visualization saved at {output_path_pca.resolve()}")
            pbar.update(1)

    except Exception as e:
        logger.exception("Error in main execution")
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

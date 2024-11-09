import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
from plotly.graph_objs import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Optional, Literal, Tuple, Union
import logging
from tqdm.auto import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import torch.backends.cudnn as cudnn
import os

# Configure logging to output to both console and file with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_visualizer_optimized.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Optimize CUDA settings for performance
cudnn.benchmark = True
cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Determine the number of workers based on CPU cores, capped at 24 for efficiency
num_workers = min(mp.cpu_count(), 24)

class HypersphericalEncoder(nn.Module):
    """
    Neural network encoder that processes market sequences and encodes them into a latent space.
    Utilizes a bidirectional GRU followed by linear projections and normalization.
    """
    def __init__(
            self,
            projection_dim: int = 128,
            sequence_length: int = 60,
            n_price_features: int = 5,
            n_indicator_features: int = 7,
            temperature: float = 0.07,
            device: torch.device = device,
            price_scaler: Optional[StandardScaler] = None,
            indicator_scaler: Optional[StandardScaler] = None
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.sequence_length = sequence_length
        self.n_price_features = n_price_features
        self.n_indicator_features = n_indicator_features
        self.temperature = temperature
        self.device = device

        self.price_scaler = price_scaler
        self.indicator_scaler = indicator_scaler

        # Bidirectional GRU to capture temporal dependencies in both directions
        self.gru = nn.GRU(
            input_size=n_price_features + n_indicator_features,
            hidden_size=projection_dim * 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        ).to(device)

        # Linear layers for projection and normalization
        self.projection1 = nn.Linear(projection_dim * 4, projection_dim * 2).to(device)
        self.layer_norm = nn.LayerNorm(projection_dim * 2).to(device)
        self.projection2 = nn.Linear(projection_dim * 2, projection_dim).to(device)

        self.eval()
        torch.set_grad_enabled(False)

    def forward(self, x: torch.Tensor, lengths: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.
        Processes input sequences and returns normalized latent vectors and GRU outputs.
        """
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            if lengths is not None:
                # Pack padded sequences for efficient processing
                packed_features = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                packed_output, gru_hidden = self.gru(packed_features)
                output, _ = pad_packed_sequence(packed_output, batch_first=True)
            else:
                output, gru_hidden = self.gru(x)

            # Concatenate the final hidden states from both directions
            gru_hidden = torch.cat((gru_hidden[-2], gru_hidden[-1]), dim=1)
            hidden = F.relu(self.layer_norm(self.projection1(gru_hidden)))
            projected = self.projection2(hidden)
            # Normalize the projected vectors to lie on a hypersphere
            normalized = F.normalize(projected / self.temperature, p=2, dim=1)

            return normalized, output

    @torch.no_grad()
    def encode_batch(self, sequences: np.ndarray, lengths: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a batch of sequences into latent vectors.
        Applies scaling and moves data to the appropriate device.
        """
        if sequences.ndim == 2:
            sequences = sequences.reshape(1, *sequences.shape)

        # Separate price and indicator features
        price_data = sequences[:, :, :self.n_price_features]
        indicator_data = sequences[:, :, self.n_price_features:self.n_price_features + self.n_indicator_features]

        if self.price_scaler is None or self.indicator_scaler is None:
            raise ValueError("Scalers not fitted")

        # Scale the data using multi-threading for efficiency
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_price = executor.submit(
                self.price_scaler.transform, 
                price_data.reshape(-1, self.n_price_features)
            )
            future_indicator = executor.submit(
                self.indicator_scaler.transform, 
                indicator_data.reshape(-1, self.n_indicator_features)
            )
            
            price_scaled = future_price.result()
            indicator_scaled = future_indicator.result()

        # Reshape the scaled data back to the original sequence dimensions
        price_scaled = price_scaled.reshape(sequences.shape[0], sequences.shape[1], -1)
        indicator_scaled = indicator_scaled.reshape(sequences.shape[0], sequences.shape[1], -1)

        # Combine price and indicator features
        combined_features = torch.from_numpy(
            np.concatenate([price_scaled, indicator_scaled], axis=-1)
        ).float().to(self.device, non_blocking=True)

        return self(combined_features, lengths)

class MarketVisualizer:
    """
    Class responsible for loading market data, preprocessing, encoding, dimensionality reduction, and visualization.
    """
    def __init__(
            self,
            data_path: str = "data/raw/btc_usdt_1m_processed.csv",
            batch_size: int = 512,
            sequence_length: int = 64,
            hidden_size: int = 128,
            device: torch.device = device,
            scaling_method: Literal["standard", "minmax"] = "standard"
    ):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.device = device
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.scaling_method = scaling_method

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info("Loading market data...")
        # Load data with specified data types for memory efficiency
        self.df = pd.read_csv(
            self.data_path,
            dtype={
                'open': np.float32,
                'high': np.float32,
                'low': np.float32,
                'close': np.float32,
                'volume': np.float32,
                'quote_asset_volume': np.float32,
                'number_of_trades': np.int32,
                'taker_buy_base_asset_volume': np.float32,
                'taker_buy_quote_asset_volume': np.float32,
                'macd': np.float32,
                'rsi_14': np.float32,
                'ema_10': np.float32
            }
        )
        
        # Convert 'open_time' to datetime and set as index
        self.df['timestamp'] = pd.to_datetime(self.df['open_time'])
        self.df.set_index('timestamp', inplace=True)

        # Define feature columns
        self.price_columns = ['open', 'high', 'low', 'close', 'volume']
        self.indicator_columns = [
            'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'macd', 'rsi_14', 'ema_10'
        ]

        # Combine price and indicator features
        self.analysis_df = self.df[self.price_columns + self.indicator_columns].copy()
        self.data_array = self.analysis_df.values

        # Initialize scalers based on the chosen scaling method
        self.price_scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
        self.indicator_scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
        
        # Fit scalers to the data
        self._fit_scalers()

        # Initialize the encoder
        self.encoder = HypersphericalEncoder(
            n_price_features=len(self.price_columns),
            n_indicator_features=len(self.indicator_columns),
            projection_dim=self.hidden_size,
            sequence_length=self.sequence_length,
            device=device,
            price_scaler=self.price_scaler,
            indicator_scaler=self.indicator_scaler
        ).to(device)

        self.all_timestamps = []

    def _fit_scalers(self):
        """
        Fits the scalers to the price and indicator data using multi-threading for efficiency.
        """
        logger.info("Fitting scalers...")
        price_data = self.data_array[:, :len(self.price_columns)]
        indicator_data = self.data_array[:, len(self.price_columns):]

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_price = executor.submit(self.price_scaler.fit, np.nan_to_num(price_data))
            future_indicator = executor.submit(self.indicator_scaler.fit, np.nan_to_num(indicator_data))
            future_price.result()
            future_indicator.result()
        logger.info("Scalers fitted successfully.")

    def visualize_hypersphere(
            self, 
            n_samples: int = 10000, 
            projection_method: str = 'pca', 
            perplexity: int = 50,
            color_feature: str = 'macd',  # New parameter for color feature
            hover_features: Optional[List[str]] = None, # New parameter for hover data
    ) -> Union[Figure, None]:
        """
        Encodes market data into latent vectors, performs dimensionality reduction, and visualizes the results.
        
        Parameters:
            n_samples (int): Number of samples to visualize.
            projection_method (str): 'pca' or 'tsne' for dimensionality reduction.
            perplexity (int): Perplexity parameter for t-SNE.
            color_feature (str): Feature name to use for color-coding the points.
            hover_features (List[str], optional): List of feature names to display on hover.
        
        Returns:
            fig (Figure): Plotly Figure object containing the visualization.
        """
        logger.info(f"Starting visualization with {n_samples} samples using {projection_method.upper()}...")
        available_sequences = len(self.analysis_df) - self.sequence_length
        max_samples = min(n_samples, available_sequences)
        if max_samples <= 0:
            logger.error("Not enough data to create sequences with the specified sequence_length.")
            raise ValueError("Not enough data to create sequences with the specified sequence_length.")

        n_batches = max(1, (max_samples + self.batch_size - 1) // self.batch_size)
        np.random.seed(42)  # For reproducibility
        start_indices = np.random.randint(0, available_sequences, size=max_samples)

        encoded_states = []
        self.all_timestamps = []

        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, max_samples)
            batch_indices = start_indices[start_idx:end_idx]

            # Efficient sequence generation using advanced NumPy indexing
            batch_sequences = self.data_array[batch_indices[:, None] + np.arange(self.sequence_length)]
            lengths = [self.sequence_length] * len(batch_sequences)  # All sequences have the same length

            try:
                # Encode the batch sequences
                batch_encoded, _ = self.encoder.encode_batch(batch_sequences, lengths=lengths)
                encoded_states.append(batch_encoded.cpu().numpy())

                # Associate each sequence with the timestamp at its end
                color_indices = batch_indices + self.sequence_length - 1
                color_indices = np.clip(color_indices, 0, len(self.df) - 1)  # Prevent out-of-bounds
                self.all_timestamps.extend(self.df.index[color_indices].tolist())
            except Exception as e:
                logger.exception(f"Error in batch {batch_idx}: {e}")
                continue

        if not encoded_states:
            logger.error("No states were successfully encoded.")
            raise ValueError("No states were successfully encoded.")

        # Concatenate all encoded states into a single array
        all_encoded = np.vstack(encoded_states)

        # Dimensionality Reduction
        if projection_method == 'pca':
            logger.info("Performing PCA dimensionality reduction...")
            reducer = PCA(n_components=3, random_state=42)
            reduced_states = reducer.fit_transform(all_encoded)
            # Prepare hover data with selected features
            if hover_features is not None:
                hover_data = {feature: self.df[feature].iloc[start_indices + self.sequence_length - 1].values for feature in hover_features}
                hover_data.update({"Timestamp": self.all_timestamps})
            else:
                hover_data = {"Timestamp": self.all_timestamps}
        elif projection_method == 'tsne':
            logger.info("Performing t-SNE dimensionality reduction...")
            reducer = TSNE(
                n_components=3,
                perplexity=min(perplexity, len(all_encoded) - 1),
                n_jobs=num_workers,
                random_state=42,
                init='random'
            )
            reduced_states = reducer.fit_transform(all_encoded)
            # Prepare hover data with selected features
            if hover_features is not None:
                hover_data = {feature: self.df[feature].iloc[start_indices + self.sequence_length - 1].values for feature in hover_features}
                hover_data.update({"Timestamp": self.all_timestamps})
            else:
                hover_data = {"Timestamp": self.all_timestamps}
        else:
            logger.error(f"Unsupported projection method: {projection_method}")
            raise ValueError(f"Unsupported projection method: {projection_method}")

        # Visualization
        if isinstance(reduced_states, np.ndarray) and reduced_states.shape[1] >= 3:
            # Extract color data based on the specified feature
            color_data = self.df[color_feature].iloc[start_indices + self.sequence_length - 1].values

            if hover_features is not None:
                hover_data = {feature: self.df[feature].iloc[start_indices + self.sequence_length - 1].values for feature in hover_features}
                hover_data.update({"Timestamp": self.all_timestamps})
            else:
                hover_data = {"Timestamp": self.all_timestamps}

            if reduced_states.shape[1] == 3:
                # Create a 3D scatter plot
                fig = px.scatter_3d(
                    x=reduced_states[:, 0],
                    y=reduced_states[:, 1],
                    z=reduced_states[:, 2],
                    color=color_data,
                    hover_data=hover_data,
                    title=f'Market State Visualization ({projection_method.upper()})',
                    labels={
                        'x': 'Component 1',
                        'y': 'Component 2',
                        'z': 'Component 3',
                        'color': color_feature
                    }
                )
            elif reduced_states.shape[1] == 2:
                # Create a 2D scatter plot
                fig = px.scatter(
                    x=reduced_states[:, 0],
                    y=reduced_states[:, 1],
                    color=color_data,
                    hover_data=hover_data,
                    title=f'Market State Visualization ({projection_method.upper()})',
                    labels={
                        'x': 'Component 1',
                        'y': 'Component 2',
                        'color': color_feature
                    }
                )
            else:
                # Unsupported number of dimensions for visualization
                fig = None
                logger.warning(f"Unsupported number of dimensions ({reduced_states.shape[1]}) for visualization.")
            
            if fig is not None:
                logger.info("Visualization created successfully.")
                return fig
            else:
                logger.warning("Visualization was not created due to unsupported dimensions.")
                return None

        logger.warning(f"Unexpected or insufficient number of dimensions for reduced states: {reduced_states.shape}")
        return None

def main():
    """
    Main function to initialize the MarketVisualizer, generate the visualization, and save it.
    """
    try:
        logger.info("Initializing MarketVisualizer...")
        viz = MarketVisualizer(
            sequence_length=128,    # Example sequence length
            hidden_size=256,        # Example hidden size
            scaling_method="minmax",
            batch_size=512
        )
        
        output_dir = Path("visualizations")
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory set to: {output_dir.resolve()}")

        # Example with t-SNE and custom color and hover features
        fig_tsne = viz.visualize_hypersphere(
            n_samples=10000,
            projection_method='tsne',
            perplexity=50,
            color_feature="close",
            hover_features=['open', 'high', 'low', 'volume']
         )

        # Example with PCA and more detailed hover information
        fig_pca = viz.visualize_hypersphere(
            n_samples=10000,
            projection_method='pca',
            color_feature='rsi_14', 
            hover_features=viz.price_columns + viz.indicator_columns  # Dynamically include all price and indicator features
        )

        # Save t-SNE visualization if it exists
        if fig_tsne:
            output_path_tsne = output_dir / "market_visualization_tsne.html"
            fig_tsne.write_html(output_path_tsne)
            logger.info(f"t-SNE Visualization saved successfully at {output_path_tsne.resolve()}")
            print(f"t-SNE Visualization saved successfully at {output_path_tsne.resolve()}")
        else:
            logger.error("t-SNE Visualization generation failed.")
            print("t-SNE Visualization generation failed.")

        # Save PCA visualization if it exists
        if fig_pca:
            output_path_pca = output_dir / "market_visualization_pca.html"
            fig_pca.write_html(output_path_pca)
            logger.info(f"PCA Visualization saved successfully at {output_path_pca.resolve()}")
            print(f"PCA Visualization saved successfully at {output_path_pca.resolve()}")
        else:
            logger.error("PCA Visualization generation failed.")
            print("PCA Visualization generation failed.")

    except Exception as e:
        logger.exception("Error in main execution")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

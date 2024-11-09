import numpy as np
import pandas as pd
import torch
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_visualizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HypersphericalEncoderFixed:
    """Fixed version of the encoder with correct feature handling"""
    def __init__(
        self,
        projection_dim: int = 128,
        sequence_length: int = 60,
        n_price_features: int = 5,
        n_indicator_features: int = 4
    ):
        self.projection_dim = projection_dim
        self.sequence_length = sequence_length
        self.n_price_features = n_price_features
        self.n_indicator_features = n_indicator_features
        
        # Initialize components
        from sklearn.preprocessing import StandardScaler
        self.price_scaler = StandardScaler()
        self.indicator_scaler = StandardScaler()
        
        # Initialize neural network components
        self.projection = torch.nn.Linear(
            n_price_features + n_indicator_features, 
            projection_dim
        )
        self.layer_norm = torch.nn.LayerNorm(projection_dim)
        
    def encode_sequence(self, sequence: np.ndarray) -> torch.Tensor:
        """Encode a sequence into the latent space"""
        # Split features
        price_data = sequence[:, :self.n_price_features]
        indicator_data = sequence[:, self.n_price_features:]
        
        # Scale features
        price_scaled = self.price_scaler.transform(price_data)
        indicator_scaled = self.indicator_scaler.transform(indicator_data)
        
        # Combine scaled features
        combined = np.concatenate([price_scaled, indicator_scaled], axis=1)
        
        # Convert to tensor and get last timestep
        combined_tensor = torch.FloatTensor(combined[-1])
        
        # Project and normalize
        projected = self.projection(combined_tensor)
        normalized = self.layer_norm(projected)
        
        return normalized

class MarketVisualizer:
    """Visualization tools for market data and encoded states"""
    
    def __init__(self, data_path: str = "data/raw/btc_usdt_1m_processed.csv"):
        """Initialize visualizer with data path"""
        logger.info(f"Initializing MarketVisualizer with data from {data_path}")
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        logger.info("Loading and preprocessing market data...")
        self.df = pd.read_csv(self.data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['open_time'])
        self.df.set_index('timestamp', inplace=True)
        
        # Define price and additional feature columns
        self.price_columns = ['open', 'high', 'low', 'close', 'volume']
        self.indicator_columns = [
            'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        
        # Convert columns to numeric
        logger.info("Converting data to numeric format...")
        all_numeric_columns = self.price_columns + self.indicator_columns
        for col in tqdm(all_numeric_columns, desc="Processing columns"):
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Create analysis dataframe
        self.analysis_df = self.df[all_numeric_columns].copy()
        logger.info(f"Analysis dataframe shape: {self.analysis_df.shape}")
        
        # Initialize encoder
        self.encoder = HypersphericalEncoderFixed(
            n_price_features=len(self.price_columns),
            n_indicator_features=len(self.indicator_columns)
        )
        
        # Fit scalers
        self._fit_scalers()
    
    def _fit_scalers(self):
        """Fit scalers on numeric data"""
        logger.info("Creating sequences for scaler fitting...")
        
        # Create sequences
        sequences = []
        n_sequences = len(self.analysis_df) - self.encoder.sequence_length + 1
        
        for i in tqdm(range(0, n_sequences, 1000), desc="Building sequences"):  # Sample every 1000th sequence
            sequence = self.analysis_df.iloc[i:i + self.encoder.sequence_length].values
            sequences.append(sequence)
        
        # Convert to numpy array
        data_for_fit = np.array(sequences)
        logger.info(f"Created {len(sequences)} sequences of shape {sequences[0].shape}")
        
        # Split data into price and indicator components
        price_data = data_for_fit[:, :, :len(self.price_columns)]
        indicator_data = data_for_fit[:, :, len(self.price_columns):]
        
        logger.info("Fitting scalers...")
        # Fit scalers
        self.encoder.price_scaler.fit(price_data.reshape(-1, len(self.price_columns)))
        self.encoder.indicator_scaler.fit(indicator_data.reshape(-1, len(self.indicator_columns)))
        
        logger.info("Scalers fitted successfully")
    
    def plot_market_overview(self, days: int = 30) -> go.Figure:
        """Create interactive overview of market data"""
        logger.info(f"Creating market overview for last {days} days")
        recent_data = self.df.tail(days * 1440)  # 1440 minutes per day
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price Action', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=recent_data.index,
                open=recent_data['open'],
                high=recent_data['high'],
                low=recent_data['low'],
                close=recent_data['close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=recent_data.index,
                y=recent_data['volume'],
                name='Volume'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Market Data Overview',
            height=800,
            xaxis2_title='Date',
            yaxis_title='Price',
            yaxis2_title='Volume',
            showlegend=True
        )
        
        logger.info("Market overview plot created successfully")
        return fig
    
    def visualize_hypersphere(
        self,
        n_samples: int = 1000,
        projection_method: str = 'pca',
        perplexity: int = 30
    ) -> go.Figure:
        """Visualize encoded states in reduced dimensionality"""
        logger.info(f"Creating hypersphere visualization using {projection_method}")
        
        # Get encoded states
        states = []
        max_samples = min(n_samples, len(self.analysis_df) - self.encoder.sequence_length)
        
        logger.info(f"Encoding {max_samples} market states...")
        for i in tqdm(range(max_samples), desc="Encoding states"):
            sequence = self.analysis_df.iloc[i:i+self.encoder.sequence_length].values
            try:
                encoded_state = self.encoder.encode_sequence(sequence)
                states.append(encoded_state)
            except Exception as e:
                logger.error(f"Error encoding sequence {i}: {str(e)}")
                continue
        
        if not states:
            raise ValueError("No states were successfully encoded")
            
        # Stack encoded states
        logger.info("Processing encoded states...")
        encoded_states = torch.stack(states).numpy()
        
        # Reduce dimensionality
        logger.info(f"Reducing dimensionality using {projection_method}...")
        if projection_method == 'pca':
            reducer = PCA(n_components=3)
        else:  # t-SNE
            reducer = TSNE(n_components=3, perplexity=perplexity)
            
        reduced_states = reducer.fit_transform(encoded_states)
        
        # Create 3D scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=reduced_states[:, 0],
                y=reduced_states[:, 1],
                z=reduced_states[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=np.arange(len(reduced_states)),  # Color by sequence order
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[f"State {i}" for i in range(len(reduced_states))],
                hoverinfo='text'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f'Encoded States Visualization ({projection_method.upper()})',
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            width=800,
            height=800
        )
        
        logger.info("Hypersphere visualization created successfully")
        return fig

def main():
    """Main visualization script"""
    logger.info("Starting visualization process...")
    
    try:
        # Initialize visualizer
        viz = MarketVisualizer()
        
        # Create output directory
        output_dir = Path("visualizations")
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Create and save visualizations
        logger.info("Creating market overview visualization...")
        market_fig = viz.plot_market_overview()
        market_fig.write_html(output_dir / "market_overview.html")
        
        logger.info("Creating hypersphere visualization...")
        hypersphere_fig_pca = viz.visualize_hypersphere(projection_method='pca')
        hypersphere_fig_pca.write_html(output_dir / "hypersphere_pca.html")
        
        logger.info("Visualizations completed successfully")
        print("\nVisualizations saved to 'visualizations' directory")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
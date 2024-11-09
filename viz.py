import numpy as np
import pandas as pd
import torch
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from train import HypersphericalEncoder, process_training_data
from market_base import MarketState
from regime import MarketRegimeDetector

class MarketVisualizer:
    """Visualization tools for market data and encoded states"""
    
    def __init__(self, data_path: str = "data/raw/btc_usdt_1m_processed.csv"):
        """
        Initialize visualizer with data path
        
        Args:
            data_path: Path to processed market data CSV
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['open_time'])
        self.df.set_index('timestamp', inplace=True)
        
        # Initialize encoder and process data
        self.results = process_training_data(str(self.data_path))
        self.encoder = self.results['encoder']
        
    def plot_market_overview(self, days: int = 30) -> go.Figure:
        """
        Create interactive overview of market data
        
        Args:
            days: Number of recent days to display
            
        Returns:
            Plotly figure with market overview
        """
        recent_data = self.df.tail(days * 1440)  # 1440 minutes per day
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price Action', 'Volume', 'Technical Indicators'),
            row_heights=[0.5, 0.2, 0.3]
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
        
        # Technical indicators
        for indicator in ['rsi_14', 'macd', 'bb_upper', 'bb_lower']:
            if indicator in recent_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=recent_data[indicator],
                        name=indicator.upper()
                    ),
                    row=3, col=1
                )
        
        fig.update_layout(
            title='Market Data Overview',
            height=1000,
            xaxis3_title='Date',
            showlegend=True
        )
        
        return fig
    
    def visualize_hypersphere(
        self,
        n_samples: int = 1000,
        projection_method: str = 'pca',
        perplexity: int = 30
    ) -> go.Figure:
        """
        Visualize encoded states in reduced dimensionality
        
        Args:
            n_samples: Number of samples to visualize
            projection_method: 'pca' or 'tsne'
            perplexity: Perplexity parameter for t-SNE
            
        Returns:
            Plotly figure with encoded states visualization
        """
        # Get encoded states
        states = []
        for i in range(min(n_samples, len(self.df) - self.encoder.sequence_length)):
            sequence = self.df.iloc[i:i+self.encoder.sequence_length].values
            timestamp = self.df.index[i+self.encoder.sequence_length-1]
            state = self.encoder.encode_sequence(sequence, timestamp)
            states.append(state)
        
        # Extract encoded tensors and regimes
        encoded_states = torch.stack([s.encoded_state for s in states]).numpy()
        regimes = [s.regime_label for s in states]
        
        # Reduce dimensionality
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
                    color=regimes,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[f"Regime: {r}" for r in regimes],
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
        
        return fig
    
    def plot_regime_transitions(self, window_days: int = 30) -> go.Figure:
        """
        Visualize regime transitions over time
        
        Args:
            window_days: Number of days to visualize
            
        Returns:
            Plotly figure with regime transitions
        """
        recent_data = self.df.tail(window_days * 1440).copy()  # 1440 minutes per day
        
        # Get regime labels for recent data
        states = []
        for i in range(len(recent_data) - self.encoder.sequence_length):
            sequence = recent_data.iloc[i:i+self.encoder.sequence_length].values
            timestamp = recent_data.index[i+self.encoder.sequence_length-1]
            state = self.encoder.encode_sequence(sequence, timestamp)
            states.append(state)
        
        regimes = [s.regime_label for s in states]
        timestamps = recent_data.index[self.encoder.sequence_length:]
        
        # Create figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Price plot
        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Regime plot
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=regimes,
                name='Regime',
                mode='lines',
                line=dict(width=2)
            ),
            row=2, col=1
        )
        
        # Add regime characteristics if available
        if hasattr(self.encoder.regime_detector, 'get_regime_characteristics'):
            chars = self.encoder.regime_detector.get_regime_characteristics()
            regime_desc = {
                r['regime_id']: r['interpretation']
                for r in chars
            }
            
            for regime_id, desc in regime_desc.items():
                fig.add_annotation(
                    x=1.02,
                    y=regime_id,
                    xref='paper',
                    yref='y2',
                    text=f"Regime {regime_id}: {desc}",
                    showarrow=False,
                    align='left'
                )
        
        fig.update_layout(
            title='Market Regimes Over Time',
            xaxis2_title='Date',
            yaxis_title='Price',
            yaxis2_title='Regime',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def visualize_regime_characteristics(self) -> go.Figure:
        """
        Create visualization of regime characteristics
        
        Returns:
            Plotly figure with regime characteristics visualization
        """
        if not hasattr(self.encoder.regime_detector, 'get_regime_characteristics'):
            raise NotImplementedError("Regime detector doesn't support characteristic analysis")
            
        chars = self.encoder.regime_detector.get_regime_characteristics()
        
        # Extract features and create DataFrame
        features = []
        for regime in chars:
            features.append({
                'Regime': f"Regime {regime['regime_id']}",
                **regime['characteristics']
            })
        
        df = pd.DataFrame(features)
        
        # Create heatmap using plotly
        fig = go.Figure(data=go.Heatmap(
            z=df.iloc[:, 1:].values,
            x=df.columns[1:],
            y=df['Regime'],
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Regime Characteristics Heatmap',
            xaxis_title='Features',
            yaxis_title='Regimes',
            height=600,
            width=1000
        )
        
        return fig

def main():
    """Main visualization script"""
    # Initialize visualizer
    viz = MarketVisualizer()
    
    # Create visualizations
    market_fig = viz.plot_market_overview()
    hypersphere_fig_pca = viz.visualize_hypersphere(projection_method='pca')
    hypersphere_fig_tsne = viz.visualize_hypersphere(projection_method='tsne')
    regime_fig = viz.plot_regime_transitions()
    characteristics_fig = viz.visualize_regime_characteristics()
    
    # Save figures
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    market_fig.write_html(output_dir / "market_overview.html")
    hypersphere_fig_pca.write_html(output_dir / "hypersphere_pca.html")
    hypersphere_fig_tsne.write_html(output_dir / "hypersphere_tsne.html")
    regime_fig.write_html(output_dir / "regime_transitions.html")
    characteristics_fig.write_html(output_dir / "regime_characteristics.html")
    
    print("Visualizations saved to 'visualizations' directory")

if __name__ == "__main__":
    main()
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from market_base import MarketState, RewardCalculator, DataValidationMixin
from regime import MarketRegimeDetector

class HypersphericalEncoder(DataValidationMixin):
    """
    Encodes market sequences into hyperspherical latent space with regime awareness
    """
    def __init__(
        self,
        projection_dim: int = 128,
        n_regimes: int = 8,
        sequence_length: int = 60,
        min_features: int = 17  # OHLCV(5) + Indicators(12)
    ):
        self.projection_dim = projection_dim
        self.n_regimes = n_regimes
        self.sequence_length = sequence_length
        self.min_features = min_features
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector(n_regimes=n_regimes)
        self.price_scaler = StandardScaler()
        self.indicator_scaler = StandardScaler()
        self.layer_norm = torch.nn.LayerNorm(projection_dim)
        
        # Initialize projection layer after feature dimension is determined
        self.projection = None
        
    def fit_scalers(self, data: np.ndarray):
        """Fits the scalers to the input data."""
        price_data = data[:, :, :5]  # OHLCV
        indicator_data = data[:, :, 5:self.min_features]  # Technical indicators
        
        # Fit scalers
        self.price_scaler.fit(price_data.reshape(-1, 5))
        self.indicator_scaler.fit(indicator_data.reshape(-1, self.min_features - 5))
    
    def _init_projection(self, total_features: int) -> None:
        """Initialize the projection layer with correct input dimension"""
        self.projection = torch.nn.Linear(total_features, self.projection_dim)
        
    def _calculate_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate market metrics from sequence data
        
        Args:
            data: Market data array of shape (batch_size, sequence_length, features)
            
        Returns:
            Dict of computed metrics
        """
        close_prices = data[:, :, 3]  # Close price is 4th column
        volume = data[:, :, 4]        # Volume is 5th column
        
        # Calculate returns and volatility
        returns = np.diff(np.log(close_prices), axis=1)
        volatility = np.std(returns, axis=1)[-1]
        
        # Volume metrics
        rel_volume = volume[:, -1] / np.mean(volume, axis=1)
        
        # Additional metrics
        price_momentum = (close_prices[:, -1] / close_prices[:, -20] - 1)[-1]
        volume_momentum = (volume[:, -1] / volume[:, -20] - 1)[-1]
        
        return {
            'volatility': float(volatility),
            'relative_volume': float(rel_volume[-1]),
            'price_momentum': float(price_momentum),
            'volume_momentum': float(volume_momentum)
        }
    
    def _encode_market_data(self, data: np.ndarray) -> torch.Tensor:
        """
        Encode raw market data into latent representation
        
        Args:
            data: Market data array of shape (batch_size, sequence_length, features)
            
        Returns:
            Encoded tensor of shape (batch_size, projection_dim)
        """
        # Split data into components
        price_data = data[:, :, :5]  # OHLCV
        indicator_data = data[:, :, 5:self.min_features]  # Technical indicators
        
        # Scale components
        price_scaled = self.price_scaler.transform(price_data.reshape(-1, 5))
        price_scaled = price_scaled.reshape(data.shape[0], self.sequence_length, 5)
        
        indicator_scaled = self.indicator_scaler.transform(indicator_data.reshape(-1, 12))
        indicator_scaled = indicator_scaled.reshape(data.shape[0], self.sequence_length, 12)
        
        # Convert to tensors
        price_tensor = torch.FloatTensor(price_scaled[:, -1, :])
        indicator_tensor = torch.FloatTensor(indicator_scaled[:, -1, :])
        
        # Initialize projection if needed
        if self.projection is None:
            total_features = price_tensor.shape[1] + indicator_tensor.shape[1]
            self._init_projection(total_features)
        
        # Combine features
        combined = torch.cat([price_tensor, indicator_tensor], dim=1)
        
        # Project and normalize
        projected = self.projection(combined)
        normalized = self.layer_norm(projected)
        
        return normalized
    
    def encode_sequence(
        self,
        data: np.ndarray,
        timestamp: pd.Timestamp
    ) -> MarketState:
        """
        Encode a single market sequence into a MarketState
        
        Args:
            data: Market sequence of shape (sequence_length, features)
            timestamp: Timestamp for the sequence
            
        Returns:
            MarketState object containing encoded state and metadata
        """
        # Validate input
        self.validate_market_data(
            data.reshape(1, *data.shape), 
            self.sequence_length, 
            self.min_features
        )
        
        # Reshape for batch processing
        data_batch = data.reshape(1, *data.shape)
        
        # Encode market data
        market_encoded = self._encode_market_data(data_batch)
        
        # Get regime
        regime = self.regime_detector.predict(
            data_batch, 
            pd.DatetimeIndex([timestamp])
        )[0]
        
        # Calculate metrics
        metrics = self._calculate_metrics(data_batch)
        
        # Create market state
        state = MarketState(
            encoded_state=market_encoded[0],
            regime_label=regime,
            current_price=float(data[-1, 3]),  # Latest close price
            timestamp=timestamp,
            metrics=metrics
        )
        
        return state
    
    def create_training_batch(
        self,
        data: pd.DataFrame,
        batch_size: int = 64
    ) -> Tuple[DataLoader, RewardCalculator]:
        """
        Create training batches and reward calculator
        
        Args:
            data: Market data DataFrame with timestamp index
            batch_size: Size of training batches
            
        Returns:
            DataLoader for training
            RewardCalculator instance
        """
        # Validate input data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        # Infer data frequency if not set
        if data.index.freq is None:
            freq = pd.infer_freq(data.index)
            if freq is None:
                raise ValueError("Cannot infer data frequency")
            data.index.freq = freq
        
        # Create sequences
        states = []
        timestamps = pd.date_range(
            start=data.index[0],
            periods=len(data)-self.sequence_length+1,
            freq=data.index.freq
        )
        
        for i in range(len(data) - self.sequence_length + 1):
            sequence = data.iloc[i:i+self.sequence_length].values
            state = self.encode_sequence(sequence, timestamps[i])
            states.append(state)
        
        # Create dataset
        encoded_states = torch.stack([s.encoded_state for s in states])
        dataset = TensorDataset(encoded_states)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        # Create reward calculator with regime-specific multipliers
        regime_multipliers = self._get_regime_multipliers()
        reward_calc = RewardCalculator(regime_multipliers=regime_multipliers)
        
        return dataloader, reward_calc
    
    def _get_regime_multipliers(self) -> Dict[int, float]:
        """Get reward multipliers based on regime characteristics"""
        if not hasattr(self.regime_detector, 'get_regime_characteristics'):
            return {}
            
        characteristics = self.regime_detector.get_regime_characteristics()
        multipliers = {}
        
        for regime in characteristics:
            regime_id = regime['regime_id']
            chars = regime['characteristics']
            
            # Calculate multiplier based on regime characteristics
            volatility = chars.get('volatility', 0)
            trend = chars.get('trend', 0)
            
            # Reduce rewards in high volatility regimes
            vol_factor = 1.0 - min(volatility, 0.5)
            
            # Increase rewards in trending regimes
            trend_factor = 1.0 + abs(trend)
            
            multipliers[regime_id] = vol_factor * trend_factor
            
        return multipliers

def process_training_data(
    csv_path: str,
    sequence_length: int = 60,
    batch_size: int = 64
) -> Dict:
    """
    Process market data for DRL training
    
    Args:
        csv_path: Path to CSV file containing market data
        sequence_length: Length of market sequences
        batch_size: Size of training batches
        
    Returns:
        Dictionary containing processed data and components
    """
    # Load and preprocess data
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['open_time'])
    df.set_index('timestamp', inplace=True)
    
    # Initialize encoder
    encoder = HypersphericalEncoder(sequence_length=sequence_length)
    
    # Prepare data for fitting scalers
    num_samples_for_fit = min(10000, len(df) - sequence_length + 1)
    data_for_fit = []
    for i in range(num_samples_for_fit):
        sequence = df.iloc[i:i+sequence_length].values
        data_for_fit.append(sequence)
    data_for_fit = np.array(data_for_fit)
    
    # Fit scalers before creating training batches
    encoder.fit_scalers(data_for_fit)
    
    # Create training batches
    dataloader, reward_calc = encoder.create_training_batch(df, batch_size)
    
    return {
        'dataloader': dataloader,
        'encoder': encoder,
        'reward_calculator': reward_calc
    }

def main():
    """Main entry point for data processing"""
    # Process training data
    results = process_training_data("data/raw/btc_usdt_1m_processed.csv")
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    torch.save(results, output_dir / "drl_training_data.pt")
    print(f"Processed data saved to {output_dir / 'drl_training_data.pt'}")

if __name__ == "__main__":
    main()
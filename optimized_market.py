import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import logging
from tqdm import tqdm
from torch.cuda import amp
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Action(Enum):
    """Trading action enumeration"""
    HOLD = 0
    BUY = 1
    SELL = 2

@dataclass
class MarketState:
    """Container for market state information with validation"""
    encoded_state: torch.Tensor
    regime_label: int
    current_price: float
    timestamp: pd.Timestamp
    metrics: Dict[str, float]
    
    def __post_init__(self):
        """Validate state attributes after initialization"""
        if not isinstance(self.encoded_state, torch.Tensor):
            raise TypeError("encoded_state must be a torch.Tensor")
        if not isinstance(self.regime_label, (int, np.integer)):
            raise TypeError("regime_label must be an integer")
        if not isinstance(self.current_price, (float, np.floating)):
            raise TypeError("current_price must be a float")
        if not isinstance(self.timestamp, pd.Timestamp):
            raise TypeError("timestamp must be a pd.Timestamp")
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dictionary")

class OptimizedMarketRegimeDetector:
    """Optimized market regime detector with GPU acceleration"""
    
    def __init__(
        self,
        n_regimes: int = 8,
        feature_window: int = 20,
        clustering_method: str = 'kmeans',
        random_state: int = 42,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.n_regimes = n_regimes
        self.feature_window = feature_window
        self.device = device
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize clustering
        self._init_clustering(clustering_method, random_state)
        
        # Batch size based on GPU memory (8GB VRAM)
        self.batch_size = 4096 if device == 'cuda' else 1024
        
        self.feature_names = [
            'volatility', 'trend', 'volume_ratio', 'momentum',
            'hour_sin', 'hour_cos', 'weekday_0', 'weekday_1', 
            'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6'
        ]
        
    def _init_clustering(self, method: str, random_state: int) -> None:
        """Initialize clustering algorithm with GPU support"""
        if method == 'kmeans':
            self.cluster_model = KMeans(
                n_clusters=self.n_regimes,
                random_state=random_state,
                n_init='auto'
            )
    
    def _encode_cyclical_time(self, hour: int) -> Tuple[float, float]:
        """Encode hour using sine and cosine transformation"""
        hour_rad = 2 * np.pi * hour / 24.0
        return np.sin(hour_rad), np.cos(hour_rad)
    
    @lru_cache(maxsize=1024)
    def _extract_features(self, data_key: Tuple[float, ...], timestamp: pd.Timestamp) -> np.ndarray:
        """Cached feature extraction for repeated sequences"""
        data = np.array(data_key).reshape(-1, 5)  # Reshape for OHLCV
        
        # Move data to GPU if available
        if self.device == 'cuda':
            data_tensor = torch.from_numpy(data).to(self.device)
            
            # Perform calculations on GPU
            close_prices = data_tensor[:, 3]
            returns = torch.diff(torch.log(close_prices))
            volatility = returns[-self.feature_window:].std().item()
            
            # Calculate trend
            x = torch.arange(self.feature_window, device=self.device, dtype=torch.float32)
            y = close_prices[-self.feature_window:]
            trend = torch.polyfit(x, y, 1)[0].item()
            
            # Move back to CPU for remaining calculations
            data = data_tensor.cpu().numpy()
        else:
            # Perform calculations on CPU
            returns = np.diff(np.log(data[:, 3]))
            volatility = np.std(returns[-self.feature_window:])
            trend = np.polyfit(np.arange(self.feature_window), data[-self.feature_window:, 3], 1)[0]
        
        volume_ratio = np.mean(data[-5:, 4]) / np.mean(data[-self.feature_window:, 4])
        momentum = data[-1, 3] / data[-self.feature_window, 3] - 1
        hour_sin, hour_cos = self._encode_cyclical_time(timestamp.hour)
        weekday_onehot = np.eye(7)[timestamp.weekday()]
        
        features = np.array([
            volatility,
            trend,
            volume_ratio,
            momentum,
            hour_sin,
            hour_cos,
            *weekday_onehot
        ])
        
        return features.reshape(1, -1)
    
    def fit(self, data: np.ndarray, timestamps: pd.DatetimeIndex) -> None:
        """Fit the regime detector on historical data"""
        # Extract features for all samples
        all_features = []
        for i in tqdm(range(len(data)), desc="Extracting features"):
            features = self._extract_features(tuple(data[i].flatten()), timestamps[i])
            all_features.append(features)
        
        features_array = np.vstack(all_features)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_array)
        
        # Fit clustering model
        self.cluster_model.fit(scaled_features)
        self.is_fitted = True
        
        # Store cluster centers
        self.cluster_centers_ = self.cluster_model.cluster_centers_
    
    def predict(self, data: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Predict market regimes for new data"""
        if not self.is_fitted:
            raise ValueError("MarketRegimeDetector must be fitted before making predictions")
        
        # Extract features
        all_features = []
        for i in range(len(data)):
            features = self._extract_features(tuple(data[i].flatten()), timestamps[i])
            all_features.append(features)
        
        features_array = np.vstack(all_features)
        
        # Scale features
        scaled_features = self.scaler.transform(features_array)
        
        # Predict regimes
        predictions = self.cluster_model.predict(scaled_features)
        
        return predictions

class OptimizedHypersphericalEncoder:
    """GPU-accelerated hyperspherical encoder with mixed precision training"""
    
    def __init__(
        self,
        projection_dim: int = 128,
        n_regimes: int = 8,
        sequence_length: int = 60,
        min_features: int = 17,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = torch.device(device)
        self.projection_dim = projection_dim
        self.sequence_length = sequence_length
        self.min_features = min_features
        
        # Initialize components
        self.regime_detector = OptimizedMarketRegimeDetector(
            n_regimes=n_regimes,
            device=device
        )
        
        # Move models to GPU
        self.projection = torch.nn.Linear(min_features, projection_dim).to(self.device)
        self.layer_norm = torch.nn.LayerNorm(projection_dim).to(self.device)
        
        # Initialize scalers
        self.price_scaler = StandardScaler()
        self.indicator_scaler = StandardScaler()
        
        # Enable automatic mixed precision
        self.scaler = amp.GradScaler()
    
    def _encode_market_data(self, data: np.ndarray) -> torch.Tensor:
        """Encode raw market data into latent representation"""
        # Split data into components
        price_data = data[:, :, :5]  # OHLCV
        indicator_data = data[:, :, 5:self.min_features]  # Technical indicators
        
        # Scale components
        price_scaled = self.price_scaler.transform(price_data.reshape(-1, 5))
        price_scaled = price_scaled.reshape(data.shape[0], self.sequence_length, 5)
        
        indicator_scaled = self.indicator_scaler.transform(indicator_data.reshape(-1, 12))
        indicator_scaled = indicator_scaled.reshape(data.shape[0], self.sequence_length, 12)
        
        # Convert to tensors and move to GPU if available
        price_tensor = torch.FloatTensor(price_scaled[:, -1, :]).to(self.device)
        indicator_tensor = torch.FloatTensor(indicator_scaled[:, -1, :]).to(self.device)
        
        # Combine features
        combined = torch.cat([price_tensor, indicator_tensor], dim=1)
        
        # Project and normalize with mixed precision
        with amp.autocast():
            projected = self.projection(combined)
            normalized = self.layer_norm(projected)
        
        return normalized
    
    def encode_sequence(
        self,
        data: Union[np.ndarray, torch.Tensor],
        timestamp: pd.Timestamp
    ) -> MarketState:
        """Encode a single market sequence into a MarketState"""
        # Convert to numpy if tensor
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
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
    
    def _calculate_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate market metrics from sequence data"""
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

class OptimizedMarketVisualizer:
    """Multi-threaded market visualizer with GPU acceleration"""
    
    def __init__(
        self,
        data_path: str,
        num_workers: int = 12,  # Optimized for Ryzen 9 7900X
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.num_workers = num_workers
        self.data_path = Path(data_path)
        
        # Load data efficiently
        self._load_data()
        
        # Initialize encoder
        self.encoder = OptimizedHypersphericalEncoder(device=device)
        
    def _load_data(self) -> None:
        """Efficient data loading with parallel processing"""
        chunks = np.array_split(pd.read_csv(self.data_path), self.num_workers)
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._process_chunk, chunks))
            
        self.df = pd.concat(results)
        self.df['timestamp'] = pd.to_datetime(self.df['open_time'])
        self.df.set_index('timestamp', inplace=True)
        
    @staticmethod
    def _process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single data chunk"""
        return chunk
    
    def visualize_parallel(self, n_samples: int = 1000) -> List[MarketState]:
        """Parallel visualization processing"""
        max_samples = len(self.df) - self.encoder.sequence_length
        n_samples = min(n_samples, max_samples)
        
        # Split work across threads
        batch_size = n_samples // self.num_workers
        batches = [(i * batch_size, min((i + 1) * batch_size, n_samples)) 
                  for i in range(self.num_workers)]
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._process_batch, start, end)
                for start, end in batches
            ]
            
            results = []
            for future in tqdm(futures, desc="Processing batches"):
                results.extend(future.result())
                
        return results
    
    def _process_batch(self, start: int, end: int) -> List[MarketState]:
        """Process a batch of visualizations"""
        results = []
        for i in range(start, end):
            sequence = self.df.iloc[i:i + self.encoder.sequence_length].values
            timestamp = self.df.index[i + self.encoder.sequence_length - 1]
            
            # Move data to GPU if available
            if self.device == 'cuda':
                sequence_tensor = torch.tensor(sequence, device='cuda')
            else:
                sequence_tensor = torch.tensor(sequence)
                
            state = self.encoder.encode_sequence(sequence_tensor, timestamp)
            results.append(state)
            
        return results

def optimize_system():
   
    if torch.cuda.is_available():
        # Set GPU memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve 20% for system
        torch.backends.cudnn.benchmark = True  # Optimize CUDA operations
        
    # Set number of threads for parallel processing
    torch.set_num_threads(24)  # Optimize for 24 logical processors
    
    # Configure numpy to use multiple threads
    np.set_num_threads(24)
    
    logger.info("System optimized for parallel processing and GPU acceleration")

class OptimizedRewardCalculator:
    """Calculate DRL rewards based on market state and actions with regime-aware adjustments"""
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        holding_cost: float = 0.0001,
        volatility_penalty: float = 0.1,
        regime_multipliers: Optional[Dict[int, float]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize reward calculator with costs and penalties"""
        self._validate_costs(transaction_cost, holding_cost, volatility_penalty)
        self.transaction_cost = transaction_cost
        self.holding_cost = holding_cost
        self.volatility_penalty = volatility_penalty
        self.device = device
        
        # Set default regime multipliers if none provided
        self.regime_multipliers = regime_multipliers or {
            0: 0.8,  # High volatility - reduce reward
            1: 1.0,  # Normal trading
            2: 1.2,  # Trending market - increase reward
        }
        
        # Initialize metrics tracking
        self.reset_metrics()
        
        # Move multipliers to GPU if available
        if self.device == 'cuda':
            self.regime_multipliers_tensor = {
                k: torch.tensor(v, device='cuda')
                for k, v in self.regime_multipliers.items()
            }
    
    def _validate_costs(
        self,
        transaction_cost: float,
        holding_cost: float,
        volatility_penalty: float
    ) -> None:
        """Validate cost parameters"""
        if not 0 <= transaction_cost <= 0.1:
            raise ValueError("transaction_cost must be between 0 and 0.1")
        if not 0 <= holding_cost <= 0.01:
            raise ValueError("holding_cost must be between 0 and 0.01")
        if not 0 <= volatility_penalty <= 1.0:
            raise ValueError("volatility_penalty must be between 0 and 1.0")
    
    def reset_metrics(self) -> None:
        """Reset accumulated metrics"""
        self.total_rewards = 0.0
        self.total_costs = 0.0
        self.total_penalties = 0.0
        self.rewards_by_regime = {}
        self.action_counts = {action: 0 for action in Action}
    
    def calculate_reward(
        self,
        current_state: MarketState,
        next_state: MarketState,
        action: Union[Action, int],
        position_size: float
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate reward for a single state transition with GPU acceleration"""
        # Validate inputs
        if isinstance(action, int):
            action = Action(action)
        if not isinstance(action, Action):
            raise ValueError(f"Invalid action: {action}")
        if not 0 <= position_size <= 1:
            raise ValueError("position_size must be between 0 and 1")
        
        # Move calculations to GPU if available
        if self.device == 'cuda':
            current_price = torch.tensor(current_state.current_price, device='cuda')
            next_price = torch.tensor(next_state.current_price, device='cuda')
            position_size_tensor = torch.tensor(position_size, device='cuda')
            
            # Calculate price change
            price_change = (next_price - current_price) / current_price
            
            # Calculate base reward
            if action == Action.BUY:
                base_reward = price_change * position_size_tensor
                costs = self.transaction_cost * position_size_tensor
            elif action == Action.SELL:
                base_reward = -price_change * position_size_tensor
                costs = self.transaction_cost * position_size_tensor
            else:  # HOLD
                base_reward = torch.tensor(0.0, device='cuda')
                costs = self.holding_cost * position_size_tensor
            
            # Apply volatility penalty
            volatility = torch.tensor(
                current_state.metrics['volatility'],
                device='cuda'
            )
            vol_penalty = -self.volatility_penalty * volatility * position_size_tensor
            
            # Get regime multiplier
            regime_multiplier = self.regime_multipliers_tensor.get(
                current_state.regime_label,
                torch.tensor(1.0, device='cuda')
            )
            
            # Calculate final reward
            reward = (base_reward - costs + vol_penalty) * regime_multiplier
            
            # Move results back to CPU
            reward = reward.item()
            metrics = {
                'base_reward': base_reward.item(),
                'costs': costs.item(),
                'vol_penalty': vol_penalty.item(),
                'regime_multiplier': regime_multiplier.item(),
                'total_reward': reward,
                'price_change': price_change.item(),
                'volatility': volatility.item()
            }
        else:
            # CPU calculations (original implementation)
            price_change = (next_state.current_price - current_state.current_price) / current_state.current_price
            
            if action == Action.BUY:
                base_reward = price_change * position_size
                costs = self.transaction_cost * position_size
            elif action == Action.SELL:
                base_reward = -price_change * position_size
                costs = self.transaction_cost * position_size
            else:  # HOLD
                base_reward = 0
                costs = self.holding_cost * position_size
            
            volatility = current_state.metrics['volatility']
            vol_penalty = -self.volatility_penalty * volatility * position_size
            
            regime_multiplier = self.regime_multipliers.get(
                current_state.regime_label,
                1.0
            )
            
            reward = (base_reward - costs + vol_penalty) * regime_multiplier
            
            metrics = {
                'base_reward': base_reward,
                'costs': costs,
                'vol_penalty': vol_penalty,
                'regime_multiplier': regime_multiplier,
                'total_reward': reward,
                'price_change': price_change,
                'volatility': volatility
            }
        
        # Update metrics
        self.total_rewards += reward
        self.total_costs += metrics['costs']
        self.total_penalties += abs(metrics['vol_penalty'])
        self.action_counts[action] += 1
        
        if current_state.regime_label not in self.rewards_by_regime:
            self.rewards_by_regime[current_state.regime_label] = []
        self.rewards_by_regime[current_state.regime_label].append(reward)
        
        return reward, metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of accumulated metrics"""
        summary = {
            'total_rewards': self.total_rewards,
            'total_costs': self.total_costs,
            'total_penalties': self.total_penalties,
            'action_counts': {action.name: count for action, count in self.action_counts.items()},
            'rewards_by_regime': {
                regime: {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'count': len(rewards)
                }
                for regime, rewards in self.rewards_by_regime.items()
            }
        }
        return summary

def main():
    """Main entry point for optimized market analysis"""
    # Configure system
    optimize_system()
    
    # Initialize components
    data_path = "data/raw/btc_usdt_1m_processed.csv"
    visualizer = OptimizedMarketVisualizer(data_path)
    
    # Process some sample data
    n_samples = 1000
    results = visualizer.visualize_parallel(n_samples)
    
    logger.info(f"Processed {len(results)} market states")
    
    # Test reward calculation
    reward_calc = OptimizedRewardCalculator()
    if len(results) >= 2:
        reward, metrics = reward_calc.calculate_reward(
            results[0],
            results[1],
            Action.BUY,
            0.5
        )
        logger.info(f"Sample reward calculation: {reward:.4f}")
        logger.info(f"Reward metrics: {metrics}")

if __name__ == "__main__":
    main()
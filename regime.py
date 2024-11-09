import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from typing import Optional, List, Union, Tuple, Dict
import pandas as pd

class MarketRegimeDetector:
    """
    Enhanced market regime detector with time-based features and flexible clustering
    """
    def __init__(
        self,
        n_regimes: int = 8,
        feature_window: int = 20,
        clustering_method: str = 'kmeans',
        random_state: int = 42
    ):
        """
        Initialize the market regime detector
        
        Args:
            n_regimes: Number of distinct market regimes to detect
            feature_window: Window size for calculating features
            clustering_method: One of ['kmeans', 'dbscan', 'gmm']
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.feature_window = feature_window
        self.random_state = random_state
        self.clustering_method = clustering_method
        
        # Initialize clustering algorithm
        if clustering_method == 'kmeans':
            self.cluster_model = KMeans(
                n_clusters=n_regimes,
                random_state=random_state,
                n_init='auto'
            )
        elif clustering_method == 'dbscan':
            self.cluster_model = DBSCAN(
                eps=0.3,
                min_samples=5
            )
        elif clustering_method == 'gmm':
            self.cluster_model = GaussianMixture(
                n_components=n_regimes,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
            
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = [
            'volatility', 'trend', 'volume_ratio', 'momentum',
            'hour_sin', 'hour_cos', 'weekday_0', 'weekday_1', 
            'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6'
        ]
    
    def _encode_cyclical_time(self, hour: int) -> Tuple[float, float]:
        """Encode hour using sine and cosine transformation"""
        hour_rad = 2 * np.pi * hour / 24.0
        return np.sin(hour_rad), np.cos(hour_rad)
    
    def _extract_features(self, data: np.ndarray, timestamp: pd.Timestamp) -> np.ndarray:
        """
        Extract market and time-based features (vectorized)
        
        Args:
            data: Market data array of shape (sequence_length, features)
            timestamp: Timestamp for time-based features
            
        Returns:
            features: Array of extracted features
        """
        # Market features (vectorized calculations)
        close_prices = data[:, 3]
        volume = data[:, 4]
        
        # Calculate returns and features
        returns = np.diff(np.log(close_prices))
        volatility = np.std(returns[-self.feature_window:])
        
        # Trend calculation
        x = np.arange(self.feature_window)
        slope, _ = np.polyfit(x, close_prices[-self.feature_window:], 1)
        
        # Volume profile
        volume_ratio = np.mean(volume[-5:]) / np.mean(volume[-self.feature_window:])
        
        # Momentum
        momentum = (close_prices[-1] / close_prices[-self.feature_window] - 1)
        
        # Time-based features
        hour_sin, hour_cos = self._encode_cyclical_time(timestamp.hour)
        weekday = timestamp.weekday()
        weekday_onehot = np.zeros(7)
        weekday_onehot[weekday] = 1
        
        # Combine all features
        features = np.concatenate([
            [volatility, slope, volume_ratio, momentum],
            [hour_sin, hour_cos],
            weekday_onehot
        ])
        
        return features.reshape(1, -1)
    
    def fit(self, data: np.ndarray, timestamps: pd.DatetimeIndex) -> None:
        """
        Fit the regime detector on historical data
        
        Args:
            data: Historical market data of shape (n_samples, sequence_length, features)
            timestamps: Corresponding timestamps for each sample
        """
        # Extract features for all samples
        all_features = []
        for i in range(len(data)):
            features = self._extract_features(data[i], timestamps[i])
            all_features.append(features)
        
        features_array = np.vstack(all_features)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_array)
        
        # Fit clustering model
        self.cluster_model.fit(scaled_features)
        self.is_fitted = True
        
        # Store cluster characteristics
        if self.clustering_method != 'dbscan':
            if hasattr(self.cluster_model, 'cluster_centers_'):
                self.cluster_centers_ = self.cluster_model.cluster_centers_
            else:
                self.cluster_centers_ = self.cluster_model.means_
    
    def predict(self, data: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Predict market regimes for new data
        
        Args:
            data: Market data of shape (n_samples, sequence_length, features)
            timestamps: Corresponding timestamps for each sample
            
        Returns:
            predictions: Array of regime labels
        """
        if not self.is_fitted:
            raise ValueError("MarketRegimeDetector must be fitted before making predictions")
        
        # Extract features
        all_features = []
        for i in range(len(data)):
            features = self._extract_features(data[i], timestamps[i])
            all_features.append(features)
        
        features_array = np.vstack(all_features)
        
        # Scale features
        scaled_features = self.scaler.transform(features_array)
        
        # Predict regimes
        if self.clustering_method in ['kmeans', 'dbscan']:
            predictions = self.cluster_model.predict(scaled_features)
        else:  # GMM
            predictions = self.cluster_model.predict(scaled_features)
        
        return predictions
    
    def get_regime_characteristics(self) -> List[Dict]:
        """
        Get characteristics of each regime based on cluster centers
        
        Returns:
            List of dictionaries containing regime characteristics
        """
        if not self.is_fitted:
            raise ValueError("MarketRegimeDetector must be fitted before getting characteristics")
        
        if self.clustering_method == 'dbscan':
            raise ValueError("Regime characteristics not available for DBSCAN")
            
        characteristics = []
        centers = self.scaler.inverse_transform(self.cluster_centers_)
        
        for i, center in enumerate(centers):
            regime_dict = {
                'regime_id': i,
                'characteristics': {
                    name: value for name, value in zip(self.feature_names, center)
                },
                'interpretation': self._interpret_regime(center)
            }
            characteristics.append(regime_dict)
            
        return characteristics
    
    def _interpret_regime(self, center: np.ndarray) -> str:
        """
        Generate human-readable interpretation of regime characteristics
        
        Args:
            center: Cluster center array
            
        Returns:
            interpretation: String describing the regime
        """
        volatility, trend, volume_ratio, momentum = center[:4]
        
        descriptions = []
        
        # Volatility interpretation
        if volatility > 1.5:
            descriptions.append("High volatility")
        elif volatility < 0.5:
            descriptions.append("Low volatility")
        
        # Trend interpretation
        if trend > 0.01:
            descriptions.append("Upward trending")
        elif trend < -0.01:
            descriptions.append("Downward trending")
        else:
            descriptions.append("Range-bound")
            
        # Volume interpretation
        if volume_ratio > 1.2:
            descriptions.append("High volume")
        elif volume_ratio < 0.8:
            descriptions.append("Low volume")
            
        # Momentum interpretation
        if momentum > 0.02:
            descriptions.append("Strong momentum")
        elif momentum < -0.02:
            descriptions.append("Weak momentum")
        
        return ", ".join(descriptions)
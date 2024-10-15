# feature_engineering.py
import cudf
import ta
from rolling_utils import calculate_rolling_statistics
import logging
import pandas as pd
from typing import Dict
from functools import lru_cache
import pywt
from sklearn.manifold import TSNE  # t-SNE is not GPU-accelerated yet
#from sklearn.preprocessing import MinMaxScaler  # Use scikit-learn for scaling if needed
from cuml.preprocessing import MinMaxScaler # Use cuML's MinMaxScaler if available
from cuml.cluster import KMeans as cuKMeans
from sklearn.cluster import KMeans as skKMeans  # Fallback to scikit-learn's KMeans
import numpy as np

# Create a dictionary to store indicator functions and their parameters
indicator_functions: Dict[str, Dict] = {
    "RSI": {"func": ta.momentum.RSIIndicator, "params": {"window": 14}},
    "MACD": {"func": ta.trend.MACD, "params": {}},  # MACD has default parameters
    "BollingerBands": {"func": ta.volatility.BollingerBands, "params": {}},
    "ATR": {"func": ta.volatility.AverageTrueRange, "params": {"window": 14}},
    "OBV": {"func": ta.volume.OnBalanceVolumeIndicator, "params": {}},
    "StochasticOscillator": {"func": ta.momentum.StochasticOscillator, "params": {}},
    "ADX": {"func": ta.trend.ADXIndicator, "params": {}},
    "CCI": {"func": ta.trend.CCIIndicator, "params": {}},
    "EMA": {"func": ta.trend.EMAIndicator, "params": {"window": 12}},
    # Add other indicators and their parameters as needed
}

@lru_cache(maxsize=None)
def calculate_wavelet_features(series: np.ndarray, wavelet: str = 'db4', level: int = 4) -> np.ndarray:
    """
    Calculates wavelet features and returns a NumPy array.
    
    Args:
        series (np.ndarray): Input data series as a NumPy array.
        wavelet (str, optional): Wavelet type. Defaults to 'db4'.
        level (int, optional): Level of decomposition. Defaults to 4.
    
    Returns:
        np.ndarray: Flattened array containing wavelet coefficients.
    """
    try:
        coeffs = pywt.wavedec(series, wavelet, level=level)
        # Flatten the coefficients into a single array:
        return np.concatenate(coeffs)
    except Exception as e:
        logging.error(f"Error in calculate_wavelet_features: {e}", exc_info=True)
        return np.array([])  # Return empty NumPy array if error

def calculate_indicators(df: cudf.DataFrame) -> cudf.DataFrame:
    """
    Calculates technical indicators and their rolling statistics, including advanced features.
    
    Args:
        df (cudf.DataFrame): Input DataFrame containing 'Close', 'High', 'Low', 'Volume' columns.
    
    Returns:
        cudf.DataFrame: DataFrame with added technical indicators and their rolling statistics.
    """
    try:
        # 1. Convert to Pandas for compatibility with `ta` and other non-cuDF operations
        df_pandas = df.to_pandas()
        close_numpy = df_pandas['Close'].values  # Get NumPy array for Wavelet and t-SNE

        close = df_pandas['Close']
        high = df_pandas['High']
        low = df_pandas['Low']
        volume = df_pandas['Volume']

        calculated_indicators = {}  # Store the calculated indicators
        for name, indicator_data in indicator_functions.items():
            indicator_func = indicator_data["func"]
            params = indicator_data["params"]
            # Apply the indicator function
            indicator = indicator_func(high=high, low=low, close=close, volume=volume, **params)
            
            # Get the primary indicator value
            try:
                main_indicator_value = getattr(indicator, name.lower())  # Try to get the attribute by lowercase name
            except AttributeError:
                main_indicator_value = getattr(indicator, name)  # If lowercase fails, try original name
            
            calculated_indicators[name] = main_indicator_value

        # 2. Wavelet Transforms
        close_wavelet_features = calculate_wavelet_features(close_numpy)
        if close_wavelet_features.size > 0:
            wavelet_columns = [f'wavelet_{i}' for i in range(close_wavelet_features.size)]
            df_pandas = pd.concat([df_pandas, pd.DataFrame(close_wavelet_features.reshape(1, -1), columns=wavelet_columns)], axis=1)  # Reshape and add to DataFrame
        else:
            logging.warning("Wavelet features calculation returned empty array.")

        # 3. Clustering (using cuML KMeans if available, otherwise fallback to scikit-learn)
        try:  # Prefer cuML if available
            kmeans = cuKMeans(n_clusters=5, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df[['Close', 'Volume']].astype('float32'))  # cuML prefers float32
        except Exception as e:  # Fallback to scikit-learn if cuML fails
            logging.warning(f"cuML KMeans failed. Falling back to scikit-learn: {e}")
            kmeans = skKMeans(n_clusters=5, random_state=42)
            df_pandas['Cluster'] = kmeans.fit_predict(df_pandas[['Close', 'Volume']].fillna(0).values)

        # 4. t-SNE Embeddings (Dimensionality reduction)
        tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='random', n_jobs=-1)  # Add n_jobs for parallelization
        tsne_embeddings = tsne.fit_transform(close_numpy.reshape(-1, 1))  # Reshape for single feature
        df_pandas[['TSNE1', 'TSNE2']] = tsne_embeddings

        # Convert back to cuDF before Fourier transforms and other cuDF operations
        df = cudf.from_pandas(df_pandas)

        # 8. Fourier Transforms
        fft_features = calculate_fft(df['Close'].fillna(0))
        df = df.merge(fft_features, left_index=True, right_index=True, how='outer')

        # 5. Rolling Statistics
        # Define core features including new advanced features
        core_features = ['Close', 'Volume', 'RSI', 'MACD', 'BollingerBands', 'ATR', 'OBV',
                         'StochasticOscillator', 'ADX', 'CCI', 'EMA',
                         'Cluster', 'TSNE1', 'TSNE2'] + wavelet_columns

        for feature in core_features:
            if feature in df.columns:
                for window in [3, 7, 14, 21]:  # Defined rolling windows
                    rolling_stats = calculate_rolling_statistics(df[feature], window)
                    rolling_stats = rolling_stats.add_prefix(f'{feature}_rolling{window}_')
                    df = df.join(rolling_stats)
            else:
                logging.warning(f"Feature '{feature}' not found in DataFrame for rolling statistics.")

        # 6. Feature Scaling using cuML MinMaxScaler if available, fallback to scikit-learn
        try:
            scaler = MinMaxScaler()
            df[core_features] = scaler.fit_transform(df[core_features].astype("float32"))  # Use cuML MinMaxScaler
        except Exception as e:
            logging.warning(f"cuML MinMaxScaler failed, falling back to scikit-learn MinMaxScaler: {e}")
            df_pandas = df.to_pandas()
            scaler = MinMaxScaler()  # Using scikit-learn
            df_pandas[core_features] = scaler.fit_transform(df_pandas[core_features].fillna(0))
            df = cudf.from_pandas(df_pandas)  # Convert back

        return df

    except Exception as e:
        logging.error(f"Error in calculate_indicators: {e}", exc_info=True)
        raise
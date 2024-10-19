# feature_engineering.py 
import ta
from functools import lru_cache
import logging
import pandas as pd
import torch
import pywt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
from typing import Dict
from logging_config import setup_logging  # Ensure logging is configured

setup_logging()

# Dictionary of Technical Indicators with their respective parameters
indicator_functions = {
    "rsi": {"func": ta.momentum.RSIIndicator, "params": {"window": 14}},
    "macd": {"func": ta.trend.MACD, "params": {}},
    "stochastic_oscillator": {"func": ta.momentum.StochasticOscillator, "params": {}},
    "bollinger_bands": {"func": ta.volatility.BollingerBands, "params": {}},
    "moving_averages": {"func": ta.volatility.BollingerBands, "params": {"window": 20}},  # Using BB for MA, adjust as needed
    "adx": {"func": ta.trend.ADXIndicator, "params": {}},
    "cci": {"func": ta.trend.CCIIndicator, "params": {}},
    "ema": {"func": None, "params": {"window": 20}},  # Marked for manual calculation
    "force_index": {"func": ta.volume.ForceIndexIndicator, "params": {"window": 13}},
    "eom": {"func": ta.volume.EaseOfMovementIndicator, "params": {"window": 14}},
    "roc": {"func": ta.momentum.ROCIndicator, "params": {"window": 12}},
    "kama": {"func": ta.momentum.KAMAIndicator, "params": {"window": 10, "pow1": 2, "pow2": 30}},
}

def calculate_ema(series: pd.Series, window: int = 20) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=None)
def calculate_wavelet_features(series: np.ndarray, wavelet: str = 'db4', level: int = 4) -> np.ndarray:
    """
    Calculates wavelet features.
    
    :param series: Input numpy array
    :param wavelet: Wavelet type (default: 'db4')
    :param level: Decomposition level (default: 4)
    :return: Wavelet features as a numpy array
    """
    try:
        if torch.cuda.is_available():
            tensor = torch.tensor(series, device='cuda:0')
            coeffs = pywt.wavedec(tensor.cpu().numpy(), wavelet, level=level)
        else:
            coeffs = pywt.wavedec(series, wavelet, level=level)
        return np.concatenate(coeffs)
    
    except Exception as e:
        logging.error(f"Error in calculate_wavelet_features: {e}", exc_info=True)
        return np.array([])

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators and their rolling statistics.
    
    :param df: Input pandas DataFrame
    :return: DataFrame with technical indicators
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    calculated_indicators = {}
    for name, indicator_data in indicator_functions.items():
        if indicator_data["func"] is not None:
            indicator = indicator_data["func"](high=high, low=low, close=close, volume=volume, **indicator_data["params"])
            calculated_indicators[name] = getattr(indicator, name, None)
        elif name == "ema":  # Manual EMA calculation
            calculated_indicators[name] = calculate_ema(close, window=indicator_functions[name]["params"]["window"])

    # Assign calculated indicators to the DataFrame
    for name, value in calculated_indicators.items():
        if value is not None:
            df[name] = value

    # Rolling Statistics (optimized)
    core_features = [
        feature for feature in [
            'Close', 'Volume', 'RSI', 'MACD', 'BollingerBands', 'ATR', 'OBV',
            'StochasticOscillator', 'ADX', 'CCI', 'EMA', 'Cluster', 'TSNE1', 'TSNE2'
        ] if feature in df.columns
    ]  # Adding conditional to prevent errors

    for window in [3, 7, 14, 21]:
        rolled = df[core_features].rolling(window)
        df = df.join(rolled.agg(['mean','std','min','max']).add_prefix(f'rolling{window}_'))  # More efficient rolling

    scaler = MinMaxScaler()
    df[core_features] = scaler.fit_transform(df[core_features].fillna(0))

    return df

def calculate_tsne_features(df: pd.DataFrame, n_components: int = 2, perplexity: int = 30) -> pd.DataFrame:
    """
    Calculates t-SNE features.
    
    :param df: Input pandas DataFrame
    :param n_components: Number of t-SNE components (default: 2)
    :param perplexity: t-SNE perplexity (default: 30)
    :return: DataFrame with t-SNE features
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_features = tsne.fit_transform(df)
    tsne_df = pd.DataFrame(tsne_features, columns=[f'TSNE_{i+1}' for i in range(n_components)])
    return tsne_df

def calculate_kmeans_features(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    Calculates K-Means clustering features.
    
    :param df: Input pandas DataFrame
    :param n_clusters: Number of K-Means clusters (default: 5)
    :return: DataFrame with K-Means cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(df)
    cluster_df = pd.DataFrame(cluster_labels, columns=['Cluster'])
    return cluster_df

def main_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering function.
    
    :param df: Input pandas DataFrame
    :return: DataFrame with engineered features
    """
    # Calculate technical indicators
    df = calculate_indicators(df)
    
    # Calculate wavelet features
    close_numpy = df['Close'].values
    close_wavelet_features = calculate_wavelet_features(close_numpy)
    
    if close_wavelet_features.size > 0:
        wavelet_columns = [f'wavelet_{i}' for i in range(close_wavelet_features.size)]
        df = pd.concat([df, pd.DataFrame(close_wavelet_features.reshape(1, -1), columns=wavelet_columns)], axis=1)
    
    # Calculate t-SNE features
    tsne_features = calculate_tsne_features(df)
    df = pd.concat([df, tsne_features], axis=1)
    
    # Calculate K-Means clustering features
    cluster_features = calculate_kmeans_features(df)
    df = pd.concat([df, cluster_features], axis=1)
    
    return df

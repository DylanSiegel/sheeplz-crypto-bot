# data_transformation.py (updated)
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from scipy.fft import fft
import logging
from typing import Optional
from logging_config import setup_logging  # Ensure logging is configured

setup_logging()

def calculate_fft(series: pd.Series) -> pd.DataFrame:
    """
    Calculates FFT features, handling NaN/infinite values.
    
    :param series: Input pandas Series
    :return: DataFrame with FFT features
    """
    series = series.dropna()
    
    # Use torch.fft on GPU if available
    if torch.cuda.is_available():
        fft_values = torch.fft.fft(torch.tensor(series.values, device='cuda:0'))
        return pd.DataFrame({
            'FFT_Real': fft_values.real.cpu().numpy(),
            'FFT_Imag': fft_values.imag.cpu().numpy(),
            'FFT_Magnitude': torch.abs(fft_values).cpu().numpy(),
            'FFT_Phase': torch.angle(fft_values).cpu().numpy()
        })
    else:
        fft_values = fft(series.values)
        return pd.DataFrame({
            'FFT_Real': np.real(fft_values),
            'FFT_Imag': np.imag(fft_values),
            'FFT_Magnitude': np.abs(fft_values),
            'FFT_Phase': np.angle(fft_values)
        })

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    :param df: Input pandas DataFrame
    :param strategy: Strategy to handle missing values (default: 'mean')
    :return: DataFrame with handled missing values
    """
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'most_frequent':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        logging.error("Invalid strategy for handling missing values. Using default 'mean' strategy.")
        return df.fillna(df.mean())

def encode_categorical_variables(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    One-Hot encode categorical variables.
    
    :param df: Input pandas DataFrame
    :param columns: List of categorical column names
    :return: DataFrame with encoded categorical variables
    """
    encoder = OneHotEncoder(sparse=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(df[columns]))
    encoded_data.columns = encoder.get_feature_names_out(columns)
    return pd.concat([df.drop(columns, axis=1), encoded_data], axis=1)

def scale_numerical_features(df: pd.DataFrame, columns: list, scaler_type: str = 'standard') -> pd.DataFrame:
    """
    Scale/Normalize numerical features.
    
    :param df: Input pandas DataFrame
    :param columns: List of numerical column names
    :param scaler_type: Type of scaler to use (default: 'standard')
    :return: DataFrame with scaled numerical features
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        logging.error("Invalid scaler type. Using default 'standard' scaler.")
        scaler = StandardScaler()
    
    df[columns] = scaler.fit_transform(df[columns])
    return df

def apply_fourier_transformation(series: pd.Series) -> pd.DataFrame:
    """
    Apply Fourier Transformation to a time series.
    
    :param series: Input pandas Series
    :return: DataFrame with Fourier Transformation features
    """
    series = series.dropna()
    
    # Use torch.fft on GPU if available
    if torch.cuda.is_available():
        fft_values = torch.fft.fft(torch.tensor(series.values, device='cuda:0'))
        return pd.DataFrame({
            'FFT_Real': fft_values.real.cpu().numpy(),
            'FFT_Imag': fft_values.imag.cpu().numpy(),
            'FFT_Magnitude': torch.abs(fft_values).cpu().numpy(),
            'FFT_Phase': torch.angle(fft_values).cpu().numpy()
        })
    else:
        fft_values = fft(series.values)
        return pd.DataFrame({
            'FFT_Real': np.real(fft_values),
            'FFT_Imag': np.imag(fft_values),
            'FFT_Magnitude': np.abs(fft_values),
            'FFT_Phase': np.angle(fft_values)
        })

def perform_pca(df: pd.DataFrame, n_components: int = 5, scaler_type: str = 'standard') -> pd.DataFrame:
    """
    Apply PCA Transformation to the DataFrame.
    
    :param df: Input pandas DataFrame
    :param n_components: Number of PCA components (default: 5)
    :param scaler_type: Type of scaler to use (default: 'standard')
    :return: DataFrame with PCA features
    """
    num_cols = df.select_dtypes(include=['number']).columns
    
    if not len(num_cols):
        return pd.DataFrame()
    
    df = scale_numerical_features(df, num_cols, scaler_type)
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df[num_cols])
    return pd.DataFrame(pca_result, columns=[f'PCA_{i+1}' for i in range(n_components)])

def main_data_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main data transformation function.
    
    :param df: Input pandas DataFrame
    :return: DataFrame with applied transformations
    """
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical variables
    categorical_cols = ['column1', 'column2']  # Replace with your categorical column names
    df = encode_categorical_variables(df, categorical_cols)
    
    # Scale numerical features
    numerical_cols = ['column3', 'column4']  # Replace with your numerical column names
    df = scale_numerical_features(df, numerical_cols)
    
    # Apply Fourier Transformation
    fourier_features = apply_fourier_transformation(df['time_series_column'])  # Replace with your time series column
    df = pd.concat([df, fourier_features], axis=1)
    
    # Apply PCA Transformation
    pca_features = perform_pca(df, n_components=5)
    df = pd.concat([df, pca_features], axis=1)
    
    return df

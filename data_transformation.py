# data_transformation.py

import cudf
import cupy as cp
from cuml.decomposition import PCA as cuPCA
from cuml.preprocessing import StandardScaler, RobustScaler
from scipy.fft import fft
import logging
from typing import Optional

def calculate_fft(series: cudf.Series) -> cudf.DataFrame:
    """
    Calculates FFT features from a given series, handling potential NaN and infinite values.
    
    Args:
        series (cudf.Series): Input data series.
    
    Returns:
        cudf.DataFrame: DataFrame containing FFT real, imaginary, magnitude, and phase components.
    """
    try:
        series = series.dropna()  # Remove NaN values before FFT
        fft_values = fft(series.values_host)  # Transfer data to host for FFT computation
        
        return cudf.DataFrame({
            'FFT_Real': cp.asnumpy(fft_values.real),      # Real part of FFT
            'FFT_Imag': cp.asnumpy(fft_values.imag),      # Imaginary part of FFT
            'FFT_Magnitude': cp.abs(fft_values),          # Magnitude of FFT
            'FFT_Phase': cp.angle(fft_values)             # Phase of FFT
        })
    except Exception as e:
        logging.error(f"Error in calculate_fft: {e}", exc_info=True)
        raise

def perform_pca(df: cudf.DataFrame, n_components: int = 5, scaler_type: str = 'standard') -> Optional[cudf.DataFrame]:
    """
    Performs Principal Component Analysis (PCA) on numerical columns with optional scaling.
    
    Args:
        df (cudf.DataFrame): Input DataFrame.
        n_components (int, optional): Number of principal components to keep. Defaults to 5.
        scaler_type (str, optional): Type of scaler to apply ('standard' or 'robust'). Defaults to 'standard'.
    
    Returns:
        Optional[cudf.DataFrame]: DataFrame containing PCA components, or empty DataFrame if no numerical columns are found.
    """
    try:
        num_cols = df.select_dtypes(include=['float64', 'float32']).columns
        if not num_cols.size:
            logging.warning("No numerical columns found for PCA.")
            return cudf.DataFrame()

        # Select scaler based on scaler_type
        if scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        scaled_data = scaler.fit_transform(df[num_cols])
        
        pca = cuPCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        return cudf.DataFrame(pca_result, columns=[f'PCA_{i+1}' for i in range(n_components)])
    except Exception as e:
        logging.error(f"Error in perform_pca: {e}", exc_info=True)
        raise

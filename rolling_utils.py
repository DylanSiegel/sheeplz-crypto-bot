# rolling_utils.py

import cudf
import logging

def calculate_rolling_statistics(series: cudf.Series, window: int) -> cudf.DataFrame:
    """
    Calculates rolling mean, standard deviation, minimum, and maximum for a given series and window.
    
    Args:
        series (cudf.Series): The input data series.
        window (int): The window size for rolling calculations.
    
    Returns:
        cudf.DataFrame: A DataFrame containing rolling 'mean', 'std', 'min', and 'max'.
    """
    try:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        rolling_min = series.rolling(window).min()
        rolling_max = series.rolling(window).max()
        
        return cudf.DataFrame({
            'mean': rolling_mean,
            'std': rolling_std,
            'min': rolling_min,
            'max': rolling_max
        })
    except Exception as e:
        logging.error(f"Error in calculate_rolling_statistics: {e}", exc_info=True)
        raise

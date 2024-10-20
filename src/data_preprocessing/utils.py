# src/data_preprocessing/utils.py

import pandas as pd
import numpy as np
import logging

def categorize_profit(returns):
    """
    Categorize returns into discrete profit/loss categories.

    Parameters:
    - returns (float): The return value to categorize.

    Returns:
    - int: The category of the return.
    """
    if returns >= 0.02:
        return 3  # Large Profit
    elif returns >= 0.01:
        return 2  # Medium Profit
    elif returns >= 0.005:
        return 1  # Small Profit
    elif returns >= -0.005:
        return 0  # Breakeven
    elif returns >= -0.01:
        return -1  # Small Loss
    elif returns >= -0.02:
        return -2  # Medium Loss
    else:
        return -3  # Large Loss

def categorize_profit_loss(df):
    """
    Categorize the target variable based on future returns for multiple horizons.
    Focuses on the 15-minute horizon for this pipeline.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'close' price.

    Returns:
    - pd.DataFrame: DataFrame with new target categories.
    """
    try:
        # Calculate returns for different horizons (assuming 15-minute intervals)
        df['target_15m'] = df['close'].shift(-1) / df['close'] - 1    # Next 15 minutes
        df['target_1h'] = df['close'].shift(-4) / df['close'] - 1     # Next 1 hour
        df['target_4h'] = df['close'].shift(-16) / df['close'] - 1    # Next 4 hours

        # Apply categorization
        df['target_category_15m'] = df['target_15m'].apply(categorize_profit)
        df['target_category_1h'] = df['target_1h'].apply(categorize_profit)
        df['target_category_4h'] = df['target_4h'].apply(categorize_profit)

        # Drop rows with NaN targets (at the end of the DataFrame)
        df.dropna(subset=['target_category_15m'], inplace=True)
        logging.info("Categorized profit/loss into discrete target categories.")
    except Exception as e:
        logging.error(f"Error during target variable categorization: {e}")
    return df

def check_data_integrity(df, filename):
    """
    Perform data integrity checks.

    Parameters:
    - df (pd.DataFrame): DataFrame to check.
    - filename (str): Name of the file being processed.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    try:
        # Check for monotonic increasing 'Open time'
        if not df.index.is_monotonic_increasing:
            logging.warning(f"'Open time' is not monotonically increasing in {filename}. Sorting the index.")
            df.sort_index(inplace=True)

        # Check for duplicate timestamps
        if df.index.duplicated().any():
            duplicates = df.index[df.index.duplicated()].unique()
            logging.warning(f"Found duplicate timestamps in {filename}: {duplicates}. Dropping duplicates.")
            df = df[~df.index.duplicated(keep='first')]

        # Additional integrity checks can be added here

        return df
    except Exception as e:
        logging.error(f"Error during data integrity checks for {filename}: {e}")
        return df

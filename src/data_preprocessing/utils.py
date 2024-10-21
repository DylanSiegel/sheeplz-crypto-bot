import pandas as pd
from loguru import logger

def categorize_profit(profit):
    try:
        if profit > 0.01:
            return 1
        elif profit < -0.01:
            return -1
        else:
            return 0
    except Exception as e:
        logger.error(f"Error categorizing profit {profit}: {e}")
        return 0

def categorize_profit_loss(df):
    try:
        logger.debug("Categorizing profit/loss into target categories...")

        timeframes = ['15m', '1h', '4h', '1d']
        for timeframe in timeframes:
            return_col = f'return_{timeframe}'
            if return_col in df.columns:
                df[f'target_category_{timeframe}'] = df[return_col].apply(categorize_profit)
            else:
                logger.warning(f"Return column {return_col} not found in DataFrame.")

        logger.debug("Categorized profit/loss successfully.")
        return df
    except Exception as e:
        logger.error(f"Error categorizing profit/loss: {e}")
        return df

def check_data_integrity(df, filename):
    try:
        logger.debug(f"Checking data integrity for file: {filename}")

        if df.index.duplicated().any():
            duplicates = df.index[df.index.duplicated()]
            logger.warning(f"Found {duplicates.sum()} duplicate timestamps in {filename}. Dropping duplicates.")
            df = df[~df.index.duplicated(keep='first')]

        critical_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = df[critical_columns].isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found in {filename}:\n{missing}")
            df.dropna(subset=critical_columns, inplace=True)
            logger.info(f"Dropped rows with missing critical values in {filename}.")

        logger.debug(f"Data integrity checks passed for {filename}.")
        return df
    except Exception as e:
        logger.error(f"Error checking data integrity for {filename}: {e}")
        return df
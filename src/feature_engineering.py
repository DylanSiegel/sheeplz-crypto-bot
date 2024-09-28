# File: src/feature_engineering.py

import talib
import pandas as pd

class FeatureEngineer:
    """
    Calculates technical indicators and other features.
    """

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds common technical indicators to the DataFrame.
        """
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['close'])
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'])

        return df

    def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds custom engineered features to the DataFrame.
        """
        # Example: Calculate percentage change
        df['pct_change'] = df['close'].pct_change()

        # Example: Calculate rolling volatility
        df['volatility'] = df['close'].rolling(window=20).std()

        # Add more custom features as needed

        return df

# Example usage
# feature_engineer = FeatureEngineer()
# df = feature_engineer.add_technical_indicators(df)
# df = feature_engineer.add_custom_features(df)
# File: feature_engineering/feature_engineer.py

import pandas as pd
import numpy as np
import talib

class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_technical_indicators(df)
        df = self.add_custom_features(df)
        return df.dropna()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.indicators.SMA.enabled:
            df['SMA'] = talib.SMA(df['close'], timeperiod=self.config.indicators.SMA.timeperiod)
        if self.config.indicators.EMA.enabled:
            df['EMA'] = talib.EMA(df['close'], timeperiod=self.config.indicators.EMA.timeperiod)
        if self.config.indicators.RSI.enabled:
            df['RSI'] = talib.RSI(df['close'], timeperiod=self.config.indicators.RSI.timeperiod)
        if self.config.indicators.MACD.enabled:
            df['MACD'], _, _ = talib.MACD(df['close'], 
                                          fastperiod=self.config.indicators.MACD.fastperiod, 
                                          slowperiod=self.config.indicators.MACD.slowperiod, 
                                          signalperiod=self.config.indicators.MACD.signalperiod)
        if self.config.indicators.ATR.enabled:
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.config.indicators.ATR.timeperiod)
        return df

    def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.custom_features.pct_change.enabled:
            df['pct_change'] = df['close'].pct_change(periods=self.config.custom_features.pct_change.window)
        if self.config.custom_features.volatility.enabled:
            df['volatility'] = df['close'].rolling(window=self.config.custom_features.volatility.window).std()
        return df
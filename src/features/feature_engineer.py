from typing import List
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccumulationDistributionIndicator
from typing import List

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Trend indicators
        df['SMA_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        macd = MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
        df['Ichimoku_a'] = ichimoku.ichimoku_a()
        df['Ichimoku_b'] = ichimoku.ichimoku_b()

        # Momentum indicators
        df['RSI'] = RSIIndicator(close=df['close']).rsi()
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['Stoch_k'] = stoch.stoch()
        df['Stoch_d'] = stoch.stoch_signal()

        # Volatility indicators
        bb = BollingerBands(close=df['close'])
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()

        # Volume indicators
        df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df['ADI'] = AccumulationDistributionIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()

        self.feature_names.extend([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
        return df

    def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=20).std() * np.sqrt(252)
        df['z_score'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['price_volume'] = df['close'] * df['volume']

        self.feature_names.extend(['log_return', 'volatility', 'z_score', 'momentum', 'price_volume'])
        return df

    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
        for feature in self.feature_names:
            for lag in lags:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
                self.feature_names.append(f'{feature}_lag_{lag}')
        return df

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_technical_indicators(df)
        df = self.add_custom_features(df)
        df = self.create_lagged_features(df)
        return df.dropna()

    def get_feature_names(self) -> List[str]:
        return self.feature_names
import pandas as pd
import talib

class FeatureEngineer:
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
        df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['close'])
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'])
        df['BBANDS_Upper'], df['BBANDS_Middle'], df['BBANDS_Lower'] = talib.BBANDS(df['close'])
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'])
        return df

    def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['pct_change'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=20).std() * np.sqrt(252)
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['price_volume'] = df['close'] * df['volume']
        return df

    def create_lagged_features(self, df: pd.DataFrame, lag: int = 5) -> pd.DataFrame:
        features = ['close', 'volume', 'SMA_20', 'RSI', 'MACD', 'ATR']
        for feature in features:
            for i in range(1, lag + 1):
                df[f'{feature}_lag_{i}'] = df[feature].shift(i)
        return df

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_technical_indicators(df)
        df = self.add_custom_features(df)
        df = self.create_lagged_features(df)
        return df.dropna()
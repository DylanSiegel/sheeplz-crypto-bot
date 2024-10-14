# feature_engineering.py
import cudf
import ta
from functools import lru_cache


def calculate_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Indicators
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd_indicator = ta.trend.MACD(close)
    macd = macd_indicator.macd()
    macd_signal = macd_indicator.macd_signal()
    bb = ta.volatility.BollingerBands(close)
    bollinger_upper = bb.bollinger_hband()
    bollinger_lower = bb.bollinger_lband()
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    return cudf.concat([
        df,
        rsi.rename('RSI'),
        macd.rename('MACD'),
        macd_signal.rename('MACD_Signal'),
        bollinger_upper.rename('Bollinger_Upper'),
        bollinger_lower.rename('Bollinger_Lower'),
        atr.rename('ATR'),
        obv.rename('OBV')
    ], axis=1)


def calculate_rolling_statistics(series, window=14):
    return cudf.DataFrame({
        'Rolling_Mean': series.rolling(window=window).mean(),
        'Rolling_Median': series.rolling(window=window).median(),
        'Rolling_Variance': series.rolling(window=window).var(),
        'Rolling_Skew': series.rolling(window=window).skew(),
        'Rolling_Kurtosis': series.rolling(window=window).kurt()
    })
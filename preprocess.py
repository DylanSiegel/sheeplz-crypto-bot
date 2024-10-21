# File: preprocess.py

import pandas as pd
import numpy as np
import os
from ta import trend, momentum, volatility, volume
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Define file paths for all timeframes
RAW_DATA_PATHS = {
    "15m": r"data\raw\btc_15m_data_2018_to_2024-2024-10-10.csv",
    "1h": r"data\raw\btc_1h_data_2018_to_2024-2024-10-10.csv",
    "4h": r"data\raw\btc_4h_data_2018_to_2024-2024-10-10.csv",
    "1d": r"data\raw\btc_1d_data_2018_to_2024-2024-10-10.csv"
}

PROCESSED_DATA_DIR = r"data\final"

# Parameters
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume'
]
TARGET_COLUMNS = {
    '15m': ['target_15m', 'target_1h', 'target_4h', 'target_1d'],
    '1h': ['target_1h', 'target_4h', 'target_1d'],
    '4h': ['target_4h', 'target_1d'],
    '1d': ['target_1d']
}
SEQ_LENGTH = 10
CHUNKSIZE = 500000  # Adjust based on memory constraints

def preprocess_chunk(df_chunk, timeframe):
    """
    Preprocesses a chunk of the dataframe by calculating technical indicators.

    Parameters:
    - df_chunk: Pandas DataFrame chunk.
    - timeframe: Timeframe identifier (e.g., '15m', '1h').

    Returns:
    - df_chunk: Processed DataFrame chunk with technical indicators.
    """
    # Ensure column names are lowercase
    df_chunk.columns = df_chunk.columns.str.lower()

    # Sort by time to ensure chronological order
    df_chunk.sort_values('open time', inplace=True)
    df_chunk.reset_index(drop=True, inplace=True)

    # Handle missing values using forward and backward fill
    df_chunk.ffill(inplace=True)
    df_chunk.bfill(inplace=True)

    # Feature Engineering
    # Returns
    df_chunk['return_15m'] = df_chunk['close'].pct_change(periods=1)
    df_chunk['return_1h'] = df_chunk['close'].pct_change(periods=4)  # 1 hour = 4 * 15m
    df_chunk['return_4h'] = df_chunk['close'].pct_change(periods=16)  # 4 hours = 16 * 15m
    df_chunk['return_1d'] = df_chunk['close'].pct_change(periods=96)  # 1 day = 96 * 15m

    # Technical Indicators
    # EMA and SMA
    df_chunk['ema_14'] = trend.EMAIndicator(close=df_chunk['close'], window=14).ema_indicator()
    df_chunk['sma_14'] = trend.SMAIndicator(close=df_chunk['close'], window=14).sma_indicator()

    # RSI
    df_chunk['rsi_14'] = momentum.RSIIndicator(close=df_chunk['close'], window=14).rsi()

    # Stochastic Oscillator
    stochastic = momentum.StochasticOscillator(
        high=df_chunk['high'],
        low=df_chunk['low'],
        close=df_chunk['close'],
        window=14,
        smooth_window=3
    )
    df_chunk['stoch_k'] = stochastic.stoch()
    df_chunk['stoch_d'] = stochastic.stoch_signal()

    # Bollinger Bands
    bollinger = volatility.BollingerBands(close=df_chunk['close'], window=20, window_dev=2)
    df_chunk['bb_mavg'] = bollinger.bollinger_mavg()
    df_chunk['bb_hband'] = bollinger.bollinger_hband()
    df_chunk['bb_lband'] = bollinger.bollinger_lband()
    df_chunk['bb_pband'] = bollinger.bollinger_pband()
    df_chunk['bb_wband'] = bollinger.bollinger_wband()

    # Keltner Channels
    keltner = volatility.KeltnerChannel(
        high=df_chunk['high'],
        low=df_chunk['low'],
        close=df_chunk['close'],
        window=20,
        window_atr=10,
        fillna=True
    )
    df_chunk['kc_hband'] = keltner.keltner_channel_hband()
    df_chunk['kc_lband'] = keltner.keltner_channel_lband()
    df_chunk['kc_mband'] = keltner.keltner_channel_mband()
    df_chunk['kc_pband'] = keltner.keltner_channel_pband()
    df_chunk['kc_wband'] = keltner.keltner_channel_wband()

    # ATR
    df_chunk['atr_14'] = volatility.AverageTrueRange(
        high=df_chunk['high'],
        low=df_chunk['low'],
        close=df_chunk['close'],
        window=14
    ).average_true_range()

    # OBV
    df_chunk['obv'] = volume.OnBalanceVolumeIndicator(
        close=df_chunk['close'],
        volume=df_chunk['volume']
    ).on_balance_volume()

    # MACD
    macd = trend.MACD(close=df_chunk['close'])
    df_chunk['macd'] = macd.macd()
    df_chunk['macd_signal'] = macd.macd_signal()
    df_chunk['macd_diff'] = macd.macd_diff()

    # ADX
    adx_indicator = trend.ADXIndicator(
        high=df_chunk['high'],
        low=df_chunk['low'],
        close=df_chunk['close'],
        window=14
    )
    df_chunk['adx'] = adx_indicator.adx()
    df_chunk['adx_pos'] = adx_indicator.adx_pos()
    df_chunk['adx_neg'] = adx_indicator.adx_neg()

    # Ulcer Index
    df_chunk['ulcer_index'] = ulcer_index(df_chunk['close'], window=14)

    # Accumulation/Distribution Indicator (ADI)
    df_chunk['adi'] = volume.AccDistIndexIndicator(
        high=df_chunk['high'],
        low=df_chunk['low'],
        close=df_chunk['close'],
        volume=df_chunk['volume']
    ).acc_dist_index()

    # Chaikin Money Flow (CMF)
    df_chunk['cmf'] = volume.ChaikinMoneyFlowIndicator(
        high=df_chunk['high'],
        low=df_chunk['low'],
        close=df_chunk['close'],
        volume=df_chunk['volume'],
        window=20
    ).chaikin_money_flow()

    # Ease of Movement (EOM)
    df_chunk['eom'] = ease_of_movement(
        high=df_chunk['high'],
        low=df_chunk['low'],
        volume=df_chunk['volume'],
        window=14
    )

    # Volume Price Trend (VPT)
    df_chunk['vpt'] = volume.VolumePriceTrendIndicator(
        close=df_chunk['close'],
        volume=df_chunk['volume']
    ).volume_price_trend()

    return df_chunk

def ulcer_index(series, window=14):
    """
    Calculates the Ulcer Index for a given series.

    Parameters:
    - series: Pandas Series of prices.
    - window: Rolling window size.

    Returns:
    - ulcer: Pandas Series of Ulcer Index values.
    """
    drawdown = series.cummax() - series
    ulcer = np.sqrt((drawdown**2).rolling(window=window).mean())
    return ulcer

def ease_of_movement(high, low, volume, window=14):
    """
    Calculates the Ease of Movement (EOM) indicator.

    Parameters:
    - high: Pandas Series of high prices.
    - low: Pandas Series of low prices.
    - volume: Pandas Series of volume.
    - window: Rolling window size.

    Returns:
    - eom: Pandas Series of EOM values.
    """
    box_ratio = ((high + low) / 2 - (high.shift(1) + low.shift(1)) / 2) / volume
    box_ratio = box_ratio.fillna(0)
    eom = box_ratio.rolling(window=window).sum()
    return eom

def add_target_variables(df, timeframe):
    """
    Adds target variables based on the timeframe.

    Parameters:
    - df: Processed DataFrame.
    - timeframe: Timeframe identifier (e.g., '15m', '1h').

    Returns:
    - df: DataFrame with added target variables.
    """
    if timeframe == '15m':
        df['target_15m'] = df['return_15m'].shift(-1)
        df['target_1h'] = df['return_1h'].shift(-4)  # 1 hour ahead
        df['target_4h'] = df['return_4h'].shift(-16)  # 4 hours ahead
        df['target_1d'] = df['return_1d'].shift(-96)  # 1 day ahead
    elif timeframe == '1h':
        df['target_1h'] = df['return_1h'].shift(-1)
        df['target_4h'] = df['return_4h'].shift(-4)  # 4 hours ahead
        df['target_1d'] = df['return_1d'].shift(-16)  # 16 hours ahead
    elif timeframe == '4h':
        df['target_4h'] = df['return_4h'].shift(-1)
        df['target_1d'] = df['return_1d'].shift(-4)  # 4 days ahead
    elif timeframe == '1d':
        df['target_1d'] = df['return_1d'].shift(-1)
    
    # Drop rows with NaN targets
    df.dropna(subset=TARGET_COLUMNS[timeframe], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def preprocess_file(timeframe, raw_path, processed_path, chunksize=CHUNKSIZE):
    """
    Processes a single raw data file and saves the processed data.

    Parameters:
    - timeframe: Timeframe identifier (e.g., '15m', '1h').
    - raw_path: Path to the raw CSV file.
    - processed_path: Path to save the processed CSV file.
    - chunksize: Number of rows per chunk.

    Returns:
    - None
    """
    processed_chunks = []
    try:
        # Read raw data in chunks
        reader = pd.read_csv(raw_path, chunksize=chunksize, parse_dates=['Open time'])

        for chunk in tqdm(reader, desc=f"Processing {timeframe} data"):
            processed_chunk = preprocess_chunk(chunk, timeframe)
            processed_chunks.append(processed_chunk)

        # Concatenate all processed chunks
        df_processed = pd.concat(processed_chunks, ignore_index=True)

        # Add target variables
        df_processed = add_target_variables(df_processed, timeframe)

        # Save the processed data with gzip compression
        df_processed.to_csv(processed_path, index=False, compression='gzip')
        print(f"Processed data for {timeframe} saved to {processed_path}")

    except Exception as e:
        print(f"Error processing {timeframe} data: {e}")

def main():
    """
    Main function to preprocess all raw data files in parallel.
    """
    # Ensure the output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    tasks = []
    for timeframe, raw_path in RAW_DATA_PATHS.items():
        processed_filename = f"processed_data_{timeframe}.csv.gz"
        processed_path = os.path.join(PROCESSED_DATA_DIR, processed_filename)
        tasks.append((timeframe, raw_path, processed_path))

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(preprocess_file, tasks)

if __name__ == "__main__":
    main()

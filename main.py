import pandas as pd
import logging
from ta import add_all_ta_features
from ta.utils import dropna
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator, CCIIndicator, PSARIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, TSIIndicator, UltimateOscillator, AwesomeOscillatorIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel, UlcerIndex
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, VolumeWeightedAveragePrice, AccDistIndexIndicator, EaseOfMovementIndicator, VolumePriceTrendIndicator, NegativeVolumeIndexIndicator
import os
import glob

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define the directory path containing the raw data files
data_dir = 'data/raw'
output_dir = 'data/processed'

# Dictionary-based window sizes for each file type
window_sizes = {
    '15m': [(3, '45m'), (12, '3h'), (48, '12h')],
    '1h': [(4, '4h'), (24, '1d'), (168, '1w')],
    '4h': [(2, '8h'), (6, '1d'), (42, '1w')],
    '1d': [(7, '1w'), (30, '1m'), (365, '1y')]
}

def apply_rolling_window_ta(df, window, suffix):
    try:
        # Trend Indicators
        df[f'SMA_{suffix}'] = SMAIndicator(df['close'], window=window).sma_indicator()
        df[f'EMA_{suffix}'] = EMAIndicator(df['close'], window=window).ema_indicator()
        macd = MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
        df[f'MACD_{suffix}'] = macd.macd()
        df[f'MACD_Signal_{suffix}'] = macd.macd_signal()
        df[f'MACD_Diff_{suffix}'] = macd.macd_diff()
        #... (other indicators)

        logger.info(f"Successfully applied rolling window TA for window {window}, suffix {suffix}")
        return df
    except Exception as e:
        logger.error(f"Error applying rolling window TA for window {window}, suffix {suffix}: {str(e)}")
        return df

def process_file(filename):
    file_type = [ft for ft in window_sizes.keys() if ft in filename][0]
    file_path = os.path.join(data_dir, filename)
    output_filename = f"{file_type}_{filename.split('.')[0]}_with_comprehensive_ta.csv"
    output_file_path = os.path.join(output_dir, output_filename)

    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Loaded file: {filename} (Shape: {df.shape})")

        # Convert 'Open time' to datetime and set as index
        df['Open time'] = pd.to_datetime(df['Open time'], errors='coerce')
        df.dropna(subset=['Open time'], inplace=True)
        df.set_index('Open time', inplace=True)
        logger.info(f"Converted 'Open time' to datetime and set as index (Shape: {df.shape})")

        # Rename columns to match ta library expectations
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        logger.info("Renamed columns to match ta library expectations")

        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in {filename}. Skipping processing.")
            return

        # Check for and handle any remaining NaN values
        if df.isna().values.any():
            logger.warning(f"NaN values detected in {filename}. Dropping rows with NaN values.")
            df = df.dropna(subset=['close'])
            logger.info(f"Dropped rows with NaN values (Shape: {df.shape})")

        # Add all TA features
        try:
            df = add_all_ta_features(
                df, 
                open="open", 
                high="high", 
                low="low", 
                close="close", 
                volume="volume",
                fillna=True
            )
            logger.info("Successfully added all TA features")
        except Exception as e:
            logger.error(f"Error adding all TA features to {filename}: {str(e)}")

        # Apply rolling window calculations
        for window, suffix in window_sizes[file_type]:
            df = apply_rolling_window_ta(df, window=window, suffix=suffix)
            if df.empty:
                logger.error(f"DataFrame became empty after applying rolling window TA for window {window}, suffix {suffix}. Skipping further processing.")
                return

        # Save the updated DataFrame to a new CSV file
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path)
        logger.info(f"Comprehensive TA features added to {filename} and saved to {output_filename}")
        logger.info("-----------------------------------")
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")

# Iterate through each file in the directory
for filename in glob.glob(os.path.join(data_dir, '*.csv')):
    filename = os.path.basename(filename)
    process_file(filename)
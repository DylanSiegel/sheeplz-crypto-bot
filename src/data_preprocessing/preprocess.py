# src/data_preprocessing/preprocess.py

import os
import glob
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator, CCIIndicator, PSARIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, TSIIndicator, UltimateOscillator, AwesomeOscillatorIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel, UlcerIndex
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, VolumeWeightedAveragePrice, AccDistIndexIndicator, EaseOfMovementIndicator, VolumePriceTrendIndicator, NegativeVolumeIndexIndicator
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Import utility functions
from .utils import categorize_profit_loss, check_data_integrity

def setup_logging():
    """
    Set up logging with both console and file handlers.
    """
    log_dir = '../../logs'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"preprocess_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_filename)),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger

logger = setup_logging()

def load_config():
    """
    Load configuration from YAML file.
    """
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Directory paths from config
DATA_DIR_REFINED = config['data']['raw_dir']
OUTPUT_DIR_FINAL = config['data']['final_dir']
OUTPUT_DIR_VISUALS = config['data']['visuals_dir']

# Preprocessing parameters
CORRELATION_THRESHOLD = config['preprocessing']['correlation_threshold']
SCALER_TYPE = config['preprocessing']['scaler_type']

OVERSAMPLE_STRATEGY = config['imbalance_handling']['oversample_strategy']
UNDERSAMPLE_STRATEGY = config['imbalance_handling']['undersample_strategy']

SEQUENCE_LENGTH = config['training']['sequence_length']
BATCH_SIZE = config['training']['batch_size']

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    """
    try:
        # Initialize indicators
        df['sma_14'] = SMAIndicator(close=df['close'], window=14).sma_indicator()
        df['ema_14'] = EMAIndicator(close=df['close'], window=14).ema_indicator()
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_diff'] = macd.macd_diff()
        df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
        df['aroon_up'] = AroonIndicator(close=df['close'], window=14).aroon_up()
        df['aroon_down'] = AroonIndicator(close=df['close'], window=14).aroon_down()
        df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
        df['psar'] = PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar()
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        df['stochastic'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14).stoch()
        df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14).williams_r()
        df['tsi'] = TSIIndicator(close=df['close'], window_slow=25, window_fast=13).tsi()
        df['ultimate_osc'] = UltimateOscillator(high=df['high'], low=df['low'], close=df['close'], window1=7, window2=14, window3=28).ultimate_oscillator()
        df['ao'] = AwesomeOscillatorIndicator(high=df['high'], low=df['low']).awesome_oscillator()
        df['bollinger_hband'] = BollingerBands(close=df['close']).bollinger_hband()
        df['bollinger_lband'] = BollingerBands(close=df['close']).bollinger_lband()
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        df['keltner_hband'] = KeltnerChannel(high=df['high'], low=df['low'], close=df['close']).keltner_channel_hband()
        df['keltner_lband'] = KeltnerChannel(high=df['high'], low=df['low'], close=df['close']).keltner_channel_lband()
        df['donchian_hband'] = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=20).donchian_channel_hband()
        df['donchian_lband'] = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=20).donchian_channel_lband()
        df['ulcer_index'] = UlcerIndex(close=df['close']).ulcer_index()
        df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index()
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df['cmf'] = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=20).chaikin_money_flow()
        df['fii'] = ForceIndexIndicator(close=df['close'], volume=df['volume'], window=13).force_index()
        df['vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price()
        df['adi'] = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()
        df['eom'] = EaseOfMovementIndicator(high=df['high'], low=df['low'], volume=df['volume']).ease_of_movement()
        df['vpt'] = VolumePriceTrendIndicator(close=df['close'], volume=df['volume']).volume_price_trend()
        df['nvi'] = NegativeVolumeIndexIndicator(close=df['close'], volume=df['volume']).negative_volume_index()
        
        # Additional indicators can be added here
        
        df.dropna(inplace=True)
        logger.info("Added technical indicators.")
        return df

def process_file(filename):
    """
    Process a single raw CSV file and save the final processed data.
    """
    try:
        logger.info(f"Processing file: {filename}")
        file_path = os.path.join(DATA_DIR_REFINED, filename)
        df = pd.read_csv(file_path, parse_dates=['Open time'])
        df.set_index('Open time', inplace=True)
        
        # Data Integrity Checks
        df = check_data_integrity(df, filename)
        
        # Remove 'Ignore' column if present
        if 'Ignore' in df.columns:
            df.drop(columns=['Ignore'], inplace=True)
            logger.info(f"Dropped 'Ignore' column from {filename}")
        
        # Drop rows with missing 'close' prices
        if df['close'].isna().any():
            initial_shape = df.shape
            df.dropna(subset=['close'], inplace=True)
            logger.info(f"Dropped rows with NaN 'close' from {filename}. Shape before: {initial_shape}, after: {df.shape}")
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Categorize target variables
        df = categorize_profit_loss(df)
        
        # Feature Engineering
        df = feature_engineering(df)
        
        # Correlation-Based Feature Selection
        df = perform_correlation_feature_selection(df, threshold=CORRELATION_THRESHOLD)
        
        # Handle Class Imbalance
        df = handle_class_imbalance(df, 'target_category_15m', OVERSAMPLE_STRATEGY, UNDERSAMPLE_STRATEGY)
        
        # Scale Features
        df = scale_features(df, 'target_category_15m', SCALER_TYPE)
        
        # Visualize Target Category Distribution
        visualize_target_category_distribution(df, filename, 'target_category_15m')
        
        # Save final data
        output_filename = f"{os.path.splitext(filename)[0]}_final.csv"
        output_file_path = os.path.join(OUTPUT_DIR_FINAL, output_filename)
        os.makedirs(OUTPUT_DIR_FINAL, exist_ok=True)
        df.to_csv(output_file_path)
        logger.info(f"Saved final refined CSV: {output_filename}")
        
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")

def visualize_target_category_distribution(df, filename, target_col):
    """
    Create and save a count plot for target category distribution.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        sns.countplot(x=target_col, data=df, palette='viridis')
        plt.title(f'Target Category Distribution - {filename}')
        plt.xlabel('Target Category')
        plt.ylabel('Count')
        output_filename = f"{os.path.splitext(filename)[0]}_target_distribution.png"
        output_file_path = os.path.join(OUTPUT_DIR_VISUALS, output_filename)
        os.makedirs(OUTPUT_DIR_VISUALS, exist_ok=True)
        plt.savefig(output_file_path)
        plt.close()
        logger.info(f"Saved target category distribution plot: {output_filename}")
    except Exception as e:
        logger.error(f"Error during visualization for {filename}: {e}")

def main():
    """
    Main function to orchestrate the data refinement pipeline.
    Utilizes multiprocessing to handle multiple files concurrently.
    """
    try:
        # Retrieve all raw CSV files
        raw_files = glob.glob(os.path.join(DATA_DIR_REFINED, '*.csv'))
        filenames = [os.path.basename(f) for f in raw_files]
        logger.info(f"Found {len(filenames)} files to process. Utilizing {multiprocessing.cpu_count()} CPU cores.")
        
        # Use ProcessPoolExecutor for parallel file processing
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            executor.map(process_file, filenames)
        
        logger.info("Completed processing all files.")
    
    except Exception as e:
        logger.error(f"Error in main processing pipeline: {e}")

if __name__ == "__main__":
    main()

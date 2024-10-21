# src/data_preprocessing/preprocess.py
import os
import sys
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from datetime import datetime
import yaml
from loguru import logger
from .indicators import add_technical_indicators

def setup_logging():
    log_dir = os.path.abspath('../../logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"preprocess_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger.remove()
    logger.add(os.path.join(log_dir, log_filename), rotation="10 MB", level="DEBUG",
               format="{time} {level} {message}")
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")

    return logger

logger = setup_logging()

def load_config():
    try:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/config.yaml'))
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.debug(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

config = load_config()

DATA_DIR_RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), config['data']['raw_dir']))
DATA_DIR_PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), config['data']['final_dir']))


def load_and_prepare_data(filename):
    try:
        logger.info(f"Processing file: {filename}")
        df = pd.read_csv(os.path.join(DATA_DIR_RAW, filename))

        # Ensure 'Open time' is parsed as datetime
        df['Open time'] = pd.to_datetime(df['Open time'], errors='coerce')
        df.set_index('Open time', inplace=True)
        df.sort_index(inplace=True)

        df.columns = [col.lower() for col in df.columns]
        df.rename(columns={'close time': 'close_time'}, inplace=True)

        columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[columns_to_keep]

        timeframe = '15m' if '15m' in filename else '1h' if '1h' in filename else '4h' if '4h' in filename else '1d'
        df['timeframe'] = timeframe

        # Convert all columns except 'timeframe' to numeric
        for col in df.columns:
            if col != 'timeframe':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN values after conversion
        df.dropna(inplace=True)

        return df
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        return None

def compute_returns(df):
    try:
        for tf in ['15m', '1h', '4h', '1d']:
            periods = {'15m': 1, '1h': 4, '4h': 16, '1d': 96}
            df[f'return_{tf}'] = df.groupby('timeframe')['close'].pct_change(periods[tf])
        return df
    except Exception as e:
        logger.error(f"Error computing returns: {e}")
        return df

def create_targets(df):
    try:
        for tf in ['15m', '1h', '4h', '1d']:
            periods = {'15m': 1, '1h': 4, '4h': 16, '1d': 96}
            df[f'future_price_{tf}'] = df.groupby('timeframe')['close'].shift(-periods[tf])
            df[f'target_{tf}'] = (df[f'future_price_{tf}'] > df['close']).astype(int)
            df.drop(f'future_price_{tf}', axis=1, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error creating target variables: {e}")
        return df

def process_data():
    try:
        all_files = glob.glob(os.path.join(DATA_DIR_RAW, "*.csv"))

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            dfs = list(executor.map(load_and_prepare_data, [os.path.basename(f) for f in all_files]))

        df = pd.concat([d for d in dfs if d is not None], axis=0)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        df = compute_returns(df)
        df = add_technical_indicators(df)
        df = create_targets(df)

        df.dropna(inplace=True)

        scaler = StandardScaler()
        features = df.columns.drop(['timeframe', 'target_15m', 'target_1h', 'target_4h', 'target_1d'])
        df[features] = scaler.fit_transform(df[features])

        os.makedirs(DATA_DIR_PROCESSED, exist_ok=True)
        for tf in df['timeframe'].unique():
            output_file = os.path.join(DATA_DIR_PROCESSED, f"processed_data_{tf}.csv")
            df[df['timeframe'] == tf].to_csv(output_file)
            logger.info(f"Processed data for {tf} timeframe saved to {output_file}")

        return df
    except Exception as e:
        logger.error(f"Error in process_data: {e}")
        raise

def main():
    try:
        logger.info("Starting data preprocessing pipeline...")
        df = process_data()
        logger.info("Data preprocessing pipeline completed successfully.")
    except Exception as e:
        logger.exception(f"Unhandled exception in main pipeline: {e}")

if __name__ == "__main__":
    main()

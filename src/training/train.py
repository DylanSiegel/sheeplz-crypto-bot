# src/training/train.py

import os
import glob
import pandas as pd
import numpy as np
import logging
from src.data_preprocessing.preprocess import (
    categorize_profit_loss,
    feature_engineering,
    perform_correlation_feature_selection,
    handle_class_imbalance,
    scale_features,
    visualize_target_category_distribution,
    generate_sequences,
    TradingDataset,
    prepare_data_for_lnn
)
from src.rl_agent.environment import TradingEnv
from src.rl_agent.agent import train_agent
from torch.utils.data import DataLoader

def setup_logging():
    """
    Set up logging with both console and file handlers.
    """
    log_dir = '../../logs'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"train_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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

# Define directories from config
DATA_DIR_FINAL = config['data']['final_dir']
MODEL_DIR = '../../models'
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    """
    Main function to orchestrate the data preparation and agent training.
    """
    try:
        # Retrieve all final CSV files
        final_files = glob.glob(os.path.join(DATA_DIR_FINAL, '*_final.csv'))
        if not final_files:
            logger.error("No final refined CSV files found. Exiting.")
            return
        
        # Select the first file for training
        selected_file = final_files[0]
        logger.info(f"Preparing environment and training with file: {selected_file}")
        df_final = pd.read_csv(selected_file, parse_dates=['Open time'])
        df_final.set_index('Open time', inplace=True)
        
        # Initialize Trading Environment
        env = TradingEnv(df_final=df_final)
        
        # Train the PPO agent
        total_timesteps = config['training']['total_timesteps']
        save_freq = config['training']['save_freq']
        train_agent(env, total_timesteps=total_timesteps, save_freq=save_freq)
        
    except Exception as e:
        logger.error(f"Error in main training pipeline: {e}")

if __name__ == "__main__":
    main()

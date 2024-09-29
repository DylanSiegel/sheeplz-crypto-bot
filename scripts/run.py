# scripts/run.py

import hydra
from omegaconf import DictConfig
import logging

from src.utils import setup_logging, get_logger
from src.data_acquisition import BinanceDataProvider
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.feature_store import FeatureStore
from src.trading import TradingExecutor
from src.rewards import ProfitReward, SharpeRatioReward
from models.trainer import TradingLitModel
from models.model import TradingModel  # Corrected import
from models.evaluator import Evaluator
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.data_loader import TradingDataset # Import the TradingDataset



@hydra.main(config_path="../config", config_name="base_config", version_base=None)
def main(cfg: DictConfig):
    # Setup logging (using Hydra's logging now)
    setup_logging(log_level=cfg.base.log_level, log_file=f"logs/trading_bot_{cfg.base.mode}.log")
    logger = get_logger()
    logger.info(f"Running in {cfg.base.mode} mode")

    # Access configurations
    api_key = cfg.exchange.api_key
    api_secret = cfg.exchange.api_secret # Added api_secret
    timeframe = cfg.exchange.timeframe
    learning_rate = cfg.model.learning_rate
    # ... other config access ...

    # Initialize Data Provider
    data_provider = BinanceDataProvider(api_key=api_key, api_secret=api_secret)
    logger.info(f"Fetching historical data for {cfg.exchange.trading_pairs} from start to end dates")

    # Example: Fetch data for the first trading pair (adapt for multiple pairs as needed)
    symbol = cfg.exchange.trading_pairs[0]
    df = data_provider.get_data(symbol=symbol, timeframe=timeframe, start_date='2023-01-01', end_date='2023-12-31')  # Use timeframe from config


    # Feature Engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.add_technical_indicators(df)
    df = feature_engineer.add_custom_features(df)
    df = df.dropna()
    logger.info("Feature engineering completed.")

    # Feature Selection
    feature_selector = FeatureSelector(threshold=cfg.features.feature_selection.threshold,
                                       max_features=cfg.features.feature_selection.max_features)
    target = (df['close'].shift(-1) > df['close']).astype(int).fillna(0).astype(int) # Define your target
    X = df[['SMA_20', 'EMA', 'RSI', 'MACD', 'ATR', 'pct_change', 'volatility']] # Features used for selection
    X_selected = feature_selector.fit_transform(X, target) # Corrected Feature Selection Usage
    logger.info(f"Selected features: {X_selected.columns.tolist()}")


    if cfg.base.mode == 'train':
        # ... (training logic - see below) ...


    elif cfg.base.mode == 'test':
        # ... (testing logic - see below) ...



# === Training Logic ===
    if cfg.base.mode == 'train':


        # Initialize Model (Using config parameters)
        model = TradingModel(input_size=X_selected.shape[1],
                             hidden_size=cfg.model.hidden_size,
                             num_layers=cfg.model.num_layers,  # Use num_layers from config
                             output_size=cfg.model.output_size,
                             dropout=cfg.model.dropout) # Add dropout

        # Initialize Trainer
        trainer = pl.Trainer(max_epochs=cfg.model.epochs,
                             gpus=1 if torch.cuda.is_available() and cfg.model.device == 'cuda' else 0, # Use device from config
                             logger=True)



        # Prepare Dataset and DataLoader (using TradingDataset)
        dataset = TradingDataset(X_selected, target)  # Create Dataset instance
        dataloader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=True)


        # Initialize Lightning Module
        lit_model = TradingLitModel(model=model,
                                    learning_rate=cfg.model.learning_rate,
                                    loss_fn=torch.nn.MSELoss(), # Define your loss function
                                    optimizer_cls=getattr(torch.optim, cfg.model.optimizer)) # Get optimizer dynamically



        # Train
        logger.info("Starting model training...")
        trainer.fit(lit_model, dataloader)  # Fit with the dataloader
        logger.info("Model training completed.")

        # Save the trained model
        torch.save(model.state_dict(), cfg.base.model_save_path + "trading_model.pth")  # Use model_save_path from base config
        logger.info(f"Model saved to {cfg.base.model_save_path}trading_model.pth")




# === Testing Logic ===

    elif cfg.base.mode == 'test':
        # Load trained model
        model = TradingModel(input_size=X_selected.shape[1],  # Corrected model initialization
                            hidden_size=cfg.model.hidden_size,
                             num_layers=cfg.model.num_layers,
                             output_size=cfg.model.output_size,
                             dropout=cfg.model.dropout)

        model.load_state_dict(torch.load(cfg.base.model_save_path + "trading_model.pth")) # Use model_save_path from config
        model.eval()
        logger.info("Trained model loaded.")

        # ... (Rest of your testing logic) ...



if __name__ == "__main__":
    main()
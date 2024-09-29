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
from src.data_loader import TradingDataset



@hydra.main(config_path="../config", config_name="base_config", version_base=None)
def main(cfg: DictConfig):
    # Setup logging (using Hydra's logging)
    setup_logging(log_level=cfg.base.log_level, log_file=f"logs/trading_bot_{cfg.base.mode}.log")
    logger = get_logger()
    logger.info(f"Running in {cfg.base.mode} mode")

    # Access configurations
    api_key = cfg.exchange.api_key
    api_secret = cfg.exchange.api_secret
    timeframe = cfg.exchange.timeframe
    learning_rate = cfg.model.learning_rate
    # ... other config access ...

    # Initialize Data Provider
    data_provider = BinanceDataProvider(api_key=api_key, api_secret=api_secret)
    logger.info(f"Fetching historical data...")  # More general message

    # Fetch data (adapt for multiple pairs if needed)
    symbol = cfg.exchange.trading_pairs[0] #  Get the first trading pair
    df = data_provider.get_data(symbol=symbol, timeframe=timeframe, start_date='2023-01-01', end_date='2023-12-31')

    # Feature Engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.add_technical_indicators(df)
    df = feature_engineer.add_custom_features(df)
    df = df.dropna()
    logger.info("Feature engineering completed.")


    # Feature Selection
    feature_selector = FeatureSelector(threshold=cfg.features.feature_selection.threshold,
                                       max_features=cfg.features.feature_selection.max_features)
    
    # Define your target variable (you'll need to adjust this based on your reward function and strategy)
    target = (df['close'].shift(-1) > df['close']).astype(int).fillna(0).astype(int)  # Example target: predict price increase/decrease
    
    # Select relevant features for feature selection (important: include the raw price data if your model needs it)
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'EMA', 'RSI', 'MACD', 'ATR', 'pct_change', 'volatility']
    X = df[feature_columns]  # These are the features you will use for training
    
    X_selected = feature_selector.fit_transform(X, target) # Perform feature selection
    logger.info(f"Selected features: {X_selected.columns.tolist()}")



    if cfg.base.mode == 'train':
        # === Training Logic ===

        # Initialize Model (Using config parameters)
        model = TradingModel(input_size=X_selected.shape[1],
                             hidden_size=cfg.model.hidden_size,
                             num_layers=cfg.model.num_layers,
                             output_size=cfg.model.output_size,
                             dropout=cfg.model.dropout)

        # Initialize Trainer
        trainer = pl.Trainer(max_epochs=cfg.model.epochs,
                             gpus=1 if torch.cuda.is_available() and cfg.model.device == 'cuda' else 0,
                             logger=True)

        # Prepare Dataset and DataLoader (using TradingDataset)
        dataset = TradingDataset(X_selected, target)
        dataloader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=True)

        # Initialize Lightning Module
        lit_model = TradingLitModel(model=model,
                                    learning_rate=cfg.model.learning_rate,
                                    loss_fn=torch.nn.MSELoss(),
                                    optimizer_cls=getattr(torch.optim, cfg.model.optimizer))

        # Train
        logger.info("Starting model training...")
        trainer.fit(lit_model, dataloader)
        logger.info("Model training completed.")

        # Save the trained model
        torch.save(model.state_dict(), cfg.base.model_save_path + "trading_model.pth")
        logger.info(f"Model saved to {cfg.base.model_save_path}trading_model.pth")



    elif cfg.base.mode == 'test':
        # === Testing Logic ===
        # Load trained model
        model = TradingModel(input_size=X_selected.shape[1],  # Use the same input size as training!
                             hidden_size=cfg.model.hidden_size,
                             num_layers=cfg.model.num_layers,
                             output_size=cfg.model.output_size,
                             dropout=cfg.model.dropout) # VERY IMPORTANT to include dropout here too!


        model.load_state_dict(torch.load(cfg.base.model_save_path + "trading_model.pth"))
        model.eval()  # Set the model to evaluation mode
        logger.info("Trained model loaded.")

        # Initialize Reward Function (Example: ProfitReward)
        reward_function = ProfitReward() # Or SharpeRatioReward(), etc.


        # Initialize Trading Executor
        trading_executor = TradingExecutor(initial_balance=cfg.trading.initial_balance,
                                           transaction_fee=cfg.trading.transaction_fee,
                                           slippage=cfg.trading.slippage)


        # Execute Backtesting (You'll likely want to use a separate dataset for testing)
        # Make sure 'df' below is your TEST data, not the training data used above.
        trade_history = trading_executor.execute_backtest(df, X_selected, target, reward_function)



        #Evaluate Performance
        evaluator = Evaluator(trade_history)
        evaluator.summary()
        evaluator.plot_equity_curve()
        evaluator.plot_drawdown()






if __name__ == "__main__":
    main()
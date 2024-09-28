# scripts/run.py

import argparse
import pandas as pd
import torch
from src.data_acquisition import BinanceDataProvider
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.trading import TradingExecutor
from src.rewards import ProfitReward, SharpeRatioReward
from models.model import TradingModel
from models.trainer import TradingLitModel, train_model
from models.evaluator import Evaluator
from src.utils import setup_logging, get_logger
import hydra
from omegaconf import DictConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Run Trading Bot")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode: train or test')
    return parser.parse_args()

@hydra.main(config_path="../config", config_name="base_config")
def main(cfg: DictConfig):
    # Setup logging
    setup_logging(log_level=cfg.base.log_level, log_file='logs/trading_bot.log')
    logger = get_logger()

    mode = cfg.get('mode', 'train')  # Can be overridden via CLI

    # Initialize Data Provider
    data_provider = BinanceDataProvider(api_key=cfg.exchange.api_key, api_secret=cfg.exchange.api_secret)
    logger.info(f"Fetching historical data for {cfg.exchange.trading_pairs} from start to end dates")

    # Example: Fetch data for the first trading pair
    symbol = cfg.exchange.trading_pairs[0]
    df = data_provider.get_data(symbol=symbol, timeframe=cfg.exchange.timeframe, start_date='2023-01-01', end_date='2023-12-31')

    # Feature Engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.add_technical_indicators(df)
    df = feature_engineer.add_custom_features(df)
    df = df.dropna()
    logger.info("Feature engineering completed.")

    # Feature Selection
    feature_selector = FeatureSelector(threshold=cfg.features.feature_selection.threshold,
                                       max_features=cfg.features.feature_selection.max_features)
    target = (df['close'].shift(-1) > df['close']).astype(int).fillna(0).astype(int)
    X = df[['SMA_20', 'EMA', 'RSI', 'MACD', 'ATR', 'pct_change', 'volatility']]
    X_selected = feature_selector.fit_transform(X, target)
    logger.info(f"Selected features: {X_selected.columns.tolist()}")

    if mode == 'train':
        # Initialize Model
        model = TradingModel(input_size=X_selected.shape[1],
                             hidden_size=cfg.model.hidden_size,
                             output_size=cfg.model.output_size)
        
        # Initialize Trainer
        trainer = pl.Trainer(max_epochs=cfg.model.epochs,
                             gpus=1 if torch.cuda.is_available() else 0,
                             logger=True)

        # Initialize Lightning Module
        lit_model = TradingLitModel(model=model,
                                    learning_rate=cfg.model.learning_rate,
                                    loss_fn=torch.nn.MSELoss(),
                                    optimizer_cls=torch.optim.Adam)

        # Prepare Dataset and DataLoader
        from models.trainer import TradingDataset
        dataset = TradingDataset(X_selected, target)
        dataloader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=True)

        # Train
        logger.info("Starting model training...")
        trainer.fit(lit_model, dataloader)
        logger.info("Model training completed.")

        # Save the trained model
        torch.save(model.state_dict(), cfg.model.model_save_path + "trading_model.pth")
        logger.info(f"Model saved to {cfg.model.model_save_path}trading_model.pth")

    elif mode == 'test':
        # Load trained model
        model = TradingModel(input_size=X_selected.shape[1],
                             hidden_size=cfg.model.hidden_size,
                             output_size=cfg.model.output_size)
        model.load_state_dict(torch.load(cfg.model.model_save_path + "trading_model.pth"))
        model.eval()
        logger.info("Trained model loaded.")

        # Initialize Reward Function
        reward_function = ProfitReward()

        # Initialize Trading Executor
        trading_executor = TradingExecutor(initial_balance=cfg.trading.initial_balance,
                                           transaction_fee=cfg.trading.transaction_fee,
                                           slippage=cfg.trading.slippage)
        trade_history = trading_executor.execute_live_trading(df, X_selected, model, reward_function)

        # Evaluate Performance
        evaluator = Evaluator(trade_history)
        evaluator.summary()
        evaluator.plot_equity_curve()
        evaluator.plot_drawdown()

if __name__ == "__main__":
    main()

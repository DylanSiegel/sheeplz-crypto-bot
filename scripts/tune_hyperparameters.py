# scripts/tune_hyperparameters.py

import argparse
import optuna
from src.models.trainer import train_model  # Corrected import
from src.data.data_acquisition import BinanceDataProvider
from src.features.feature_engineer import FeatureEngineer  # Corrected import
from src.features.feature_selector import FeatureSelector
from src.trading.trading_executor import TradingExecutor 
from src.models.evaluator import Evaluator # Corrected import
from src.rewards.rewards import ProfitReward, SharpeRatioReward
from src.utils.utils import setup_logging, get_logger # Corrected import
import pandas as pd
from src.models.lstm_model import TradingModel # Corrected import (Assuming it's in lstm_model.py)
from src.models.trainer import TradingLitModel 
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning with Optuna")
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Candlestick timeframe')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    return parser.parse_args()

def objective(trial, cfg):
    # Initialize Data Provider
    data_provider = BinanceDataProvider(api_key=cfg.exchange.api_key, api_secret=cfg.exchange.api_secret)
    df = data_provider.get_data(symbol=cfg.exchange.trading_pairs[0], timeframe=cfg.exchange.timeframe,
                                start_date=cfg.start_date, end_date=cfg.end_date)

    # Feature Engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.add_technical_indicators(df)
    df = feature_engineer.add_custom_features(df)
    df = df.dropna()

    # Feature Selection
    feature_selector = FeatureSelector(threshold=0.01, max_features=10)
    target = (df['close'].shift(-1) > df['close']).astype(int).fillna(0).astype(int)
    X = df[['SMA_20', 'EMA', 'RSI', 'MACD', 'ATR', 'pct_change', 'volatility']]
    X_selected = feature_selector.fit_transform(X, target)

    # Define hyperparameters to tune
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_layers = trial.suggest_int('num_layers', 1, 4)

    # Initialize Model
    model = TradingModel(input_size=X_selected.shape[1],
                         hidden_size=hidden_size,
                         output_size=3)

    # Initialize Lightning Module
    lit_model = TradingLitModel(model=model,
                                learning_rate=learning_rate,
                                loss_fn=torch.nn.MSELoss(),
                                optimizer_cls=torch.optim.Adam)

    # Prepare Dataset and DataLoader
    from models.trainer import TradingDataset
    dataset = TradingDataset(X_selected, target)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize Trainer
    trainer = pl.Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0, logger=False)

    # Train
    trainer.fit(lit_model, dataloader)

    # Evaluate on validation set (split data accordingly)
    # For simplicity, using the same data
    preds = model(torch.tensor(X_selected.values, dtype=torch.float32))
    preds = torch.argmax(preds, dim=1).numpy()
    accuracy = (preds == target.values).mean()

    return accuracy

def main():
    # Setup logging
    setup_logging()
    logger = get_logger()

    args = parse_args()

    # Configuration dictionary (could be loaded from a file or defined here)
    cfg = {
        'exchange': {
            'api_key': 'YOUR_API_KEY',
            'api_secret': 'YOUR_API_SECRET',
            'trading_pairs': [args.symbol],
            'timeframe': args.timeframe
        },
        'start_date': args.start_date,
        'end_date': args.end_date
    }

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, cfg), n_trials=50)

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save study results
    study.trials_dataframe().to_csv("optuna_study_results.csv")
    logger.info("Study results saved to optuna_study_results.csv")

if __name__ == "__main__":
    main()

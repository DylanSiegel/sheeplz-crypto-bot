# File: src/main.py

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from src.data_acquisition import BinanceDataProvider
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.feature_store import FeatureStore
from src.trading import TradingExecutor
from src.rewards import ProfitReward, SharpeRatioReward
from models.trainer import TradingLitModel
from models.model import TradingModel
from models.evaluator import Evaluator
from src.utils import setup_logging, get_logger
import pandas as pd

@hydra.main(config_path="../config", config_name="base_config")
def main(cfg: DictConfig):
    # Setup logging
    setup_logging(log_level=cfg.base.log_level, log_file='logs/trading_bot.log')
    logger = get_logger()

    # Initialize Data Provider
    data_provider = BinanceDataProvider(api_key=cfg.exchange.api_key, api_secret=cfg.exchange.api_secret)
    logger.info(f"Fetching historical data for {cfg.exchange.trading_pairs} from start to end dates")

    # Fetch data for each trading pair
    all_trade_history = []
    for symbol in cfg.exchange.trading_pairs:
        df = data_provider.get_data(
            symbol=symbol,
            timeframe=cfg.exchange.timeframe,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        all_trade_history.append(df)

    # Combine data if multiple trading pairs
    if len(all_trade_history) > 1:
        df = pd.concat(all_trade_history, keys=cfg.exchange.trading_pairs, names=['symbol', 'index']).reset_index(level=0)
    else:
        df = all_trade_history[0]

    # Feature Engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.add_technical_indicators(df)
    df = feature_engineer.add_custom_features(df)
    df = df.dropna()
    logger.info("Feature engineering completed.")

    # Feature Selection
    feature_selector = FeatureSelector(
        method=cfg.features.feature_selection.method,
        threshold=cfg.features.feature_selection.threshold,
        max_features=cfg.features.feature_selection.max_features
    )
    target = (df['close'].shift(-1) > df['close']).astype(int).fillna(0).astype(int)
    feature_columns = ['SMA_20', 'EMA', 'RSI', 'MACD', 'ATR', 'pct_change', 'volatility']
    X = df[feature_columns]
    X_selected = feature_selector.fit_transform(X, target)
    selected_features = feature_selector.get_selected_features()
    logger.info(f"Selected features: {selected_features}")

    # Save selected features
    feature_store = FeatureStore(feature_save_path=cfg.base.feature_save_path)
    feature_store.save_features(X_selected, "selected_features.csv")
    logger.info("Selected features saved.")

    # Initialize Dataset and DataLoader
    from src.data_loader import TradingDataset
    dataset = TradingDataset(X_selected, target)
    dataloader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=True)

    # Initialize Model
    model = TradingModel(
        input_size=X_selected.shape[1],
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        output_size=cfg.model.output_size,
        dropout=cfg.model.dropout
    )

    # Initialize Lightning Module
    lit_model = TradingLitModel(
        model=model,
        learning_rate=cfg.model.learning_rate,
        loss_fn=torch.nn.MSELoss(),
        optimizer_cls=torch.optim.Adam
    )

    # Initialize Trainer
    import pytorch_lightning as pl
    trainer = pl.Trainer(
        max_epochs=cfg.model.epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=True
    )

    # Train the model
    logger.info("Starting model training...")
    trainer.fit(lit_model, dataloader)
    logger.info("Model training completed.")

    # Save the trained model
    torch.save(model.state_dict(), cfg.model.model_save_path + "trading_model.pth")
    logger.info(f"Model saved to {cfg.model.model_save_path}trading_model.pth")

    # Initialize Reward Function
    if cfg.trading.reward_type == 'profit':
        reward_function = ProfitReward()
    else:
        reward_function = SharpeRatioReward()

    # Initialize Trading Executor
    trading_executor = TradingExecutor(
        initial_balance=cfg.trading.initial_balance,
        transaction_fee=cfg.trading.transaction_fee,
        slippage=cfg.trading.slippage
    )

    # Execute Backtest
    trade_history = trading_executor.execute_backtest(df, X_selected, target, reward_function)
    logger.info("Backtest execution completed.")

    # Evaluate Performance
    evaluator = Evaluator(trade_history)
    evaluator.summary()
    evaluator.plot_equity_curve()
    evaluator.plot_drawdown()

    # Optionally, track experiment with MLflow or other tools
    if cfg.base.experiment_tracking.enabled and cfg.base.experiment_tracking.tool.lower() == 'mlflow':
        import mlflow
        mlflow.start_run()
        mlflow.log_params(cfg)
        mlflow.log_metrics({
            "Sharpe Ratio": evaluator.calculate_sharpe_ratio(),
            "Max Drawdown": evaluator.calculate_max_drawdown(),
            "Total Return": evaluator.calculate_total_return()
        })
        mlflow.end_run()
        logger.info("Experiment logged with MLflow.")

if __name__ == "__main__":
    main()

import hydra
from omegaconf import DictConfig
from agents.agent_manager import AgentManager
from environments.crypto_trading_env import CryptoTradingEnv
from src.data.data_acquisition import BinanceDataProvider
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from utils.utils import setup_logging, get_logger
import pandas as pd

@hydra.main(config_path="../config", config_name="base_config", version_base=None)
def main(cfg: DictConfig):
    setup_logging(log_level=cfg.logging.log_level, log_file=f"logs/trading_bot_{cfg.base.mode}.log")
    logger = get_logger()
    logger.info(f"Running in {cfg.base.mode} mode")

    # Initialize Data Provider and fetch data
    data_provider = BinanceDataProvider(api_key=cfg.exchange.api_key, api_secret=cfg.exchange.api_secret)
    data = {}
    for symbol in cfg.exchange.trading_pairs:
        df = data_provider.get_data(symbol=symbol, timeframe=cfg.exchange.timeframe, start_date=cfg.data.start_date, end_date=cfg.data.end_date)
        data[symbol] = df

    # Feature Engineering and Selection
    feature_engineer = FeatureEngineer()
    feature_selector = FeatureSelector(threshold=cfg.features.feature_selection.threshold, max_features=cfg.features.feature_selection.max_features)
    
    processed_data = {}
    for symbol, df in data.items():
        df = feature_engineer.process_features(df)
        X = df[cfg.features.feature_columns]
        y = (df['close'].shift(-1) > df['close']).astype(int).fillna(0)
        X_selected = feature_selector.fit_transform(X, y)
        processed_data[symbol] = (X_selected, y)

    # Initialize environments for each timeframe
    environments = {
        timeframe: CryptoTradingEnv(
            data=data[cfg.exchange.trading_pairs[0]],  # Using the first trading pair for simplicity
            processed_features=processed_data[cfg.exchange.trading_pairs[0]][0],
            reward_function=cfg.trading.reward_function,
            **cfg.environment
        ) for timeframe in cfg.trading.timeframes
    }

    # Initialize AgentManager
    agent_manager = AgentManager(agent_configs=cfg.agents, environments=environments)

    if cfg.base.mode == 'train':
        results = agent_manager.train_agents(num_episodes=cfg.training.num_episodes)
        logger.info(f"Training completed. Results: {results}")
        agent_manager.save_agents(cfg.paths.model_save_path)
    elif cfg.base.mode == 'test':
        agent_manager.load_agents(cfg.paths.model_save_path)
        evaluation_results = agent_manager.evaluate_agents(num_episodes=cfg.evaluation.num_episodes)
        logger.info(f"Evaluation completed. Results: {evaluation_results}")

if __name__ == "__main__":
    main()
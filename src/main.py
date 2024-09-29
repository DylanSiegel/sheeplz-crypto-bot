# File: main.py

import hydra
from omegaconf import DictConfig
import ray
from agents.agent_manager import AgentManager
from environments.crypto_trading_env import CryptoTradingEnv
from models.trading_model import TradingModel
from utils.logging_config import setup_logging, get_logger
from data.data_provider import BinanceDataProvider
from feature_engineering.feature_engineer import FeatureEngineer
from feature_engineering.feature_selector import FeatureSelector
from rewards.reward_functions import ProfitReward, SharpeRatioReward
import torch

@hydra.main(config_path="config", config_name="base_config")
def main(cfg: DictConfig):
    # Setup logging
    setup_logging(cfg.logging)
    logger = get_logger(__name__)

    # Initialize Ray
    ray.init(num_cpus=cfg.ray.num_cpus, num_gpus=cfg.ray.num_gpus)

    # Initialize Data Provider
    data_provider = BinanceDataProvider(api_key=cfg.exchange.api_key, api_secret=cfg.exchange.api_secret)
    
    # Fetch data for each trading pair
    data = {}
    for symbol in cfg.exchange.trading_pairs:
        df = data_provider.get_data(
            symbol=symbol,
            timeframe=cfg.exchange.timeframe,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date
        )
        data[symbol] = df
        logger.info(f"Fetched {len(df)} rows for {symbol}")

    # Feature Engineering
    feature_engineer = FeatureEngineer(cfg.features)
    feature_selector = FeatureSelector(cfg.feature_selection)
    
    processed_data = {}
    for symbol, df in data.items():
        df = feature_engineer.process_features(df)
        X = df[cfg.features.feature_columns]
        y = (df['close'].shift(-1) > df['close']).astype(int).fillna(0)
        X_selected = feature_selector.fit_transform(X, y)
        processed_data[symbol] = (X_selected, y)
        logger.info(f"Processed features for {symbol}. Selected features: {X_selected.columns.tolist()}")

    # Initialize environments and agents
    environments = {
        symbol: CryptoTradingEnv(
            data=data,
            processed_features=X,
            reward_function=ProfitReward() if cfg.trading.reward_type == 'profit' else SharpeRatioReward(),
            **cfg.environment
        ) for symbol, (X, _) in processed_data.items()
    }

    agent_configs = [
        {
            "model": TradingModel(
                input_size=next(iter(processed_data.values()))[0].shape[1],
                **cfg.model
            ),
            "reward_function": ProfitReward() if cfg.trading.reward_type == 'profit' else SharpeRatioReward(),
            "agent_id": f"agent_{i}"
        } for i in range(cfg.agents.num_agents)
    ]

    agent_manager = AgentManager(environments, agent_configs)

    # Train agents
    results = agent_manager.train_agents(num_episodes=cfg.training.num_episodes)
    logger.info(f"Training completed. Results: {results}")

    # Evaluate agents
    evaluation_results = agent_manager.evaluate_agents(num_episodes=cfg.evaluation.num_episodes)
    logger.info(f"Evaluation completed. Results: {evaluation_results}")

    # Save trained agents
    agent_manager.save_agents(cfg.paths.model_save_path)

    ray.shutdown()

if __name__ == "__main__":
    main()
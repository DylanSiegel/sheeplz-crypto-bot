import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
import numpy as np
from typing import Dict
import time
from src.data.data_acquisition import BinanceDataProvider
from src.features.feature_engineer import FeatureEngineer
from src.models.lstm_model import TradingModel
from src.agents.agent_manager import AgentManager
from src.environments.crypto_trading_env import CryptoTradingEnv
from src.rewards.rewards import get_reward_function
from utils.utils import setup_logging, get_logger
from dotenv import load_dotenv  # For loading environment variables
import os
from src.visualization.visualization import Visualization # Import for visualizations

logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../config/secrets.env'))

def preprocess_data(data: pd.DataFrame, feature_engineer: FeatureEngineer) -> pd.DataFrame:
    processed_data = feature_engineer.process_features(data)
    return processed_data.fillna(0)  # Fill NaN values with 0 to avoid issues

def create_environment(data: pd.DataFrame, config: DictConfig) -> CryptoTradingEnv:
    reward_function = get_reward_function(config.trading.reward_function)
    return CryptoTradingEnv(data, initial_balance=config.trading.initial_balance, 
                            transaction_fee=config.trading.transaction_fee, 
                            reward_function=reward_function)

def train_model(model: TradingModel, env: CryptoTradingEnv, config: DictConfig, visualization: Visualization):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    criterion = torch.nn.MSELoss()
    training_data = [] # Initialize for training visualization

    for episode in range(config.training.num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        actions_taken = [] # Initialize for tracking actions in the episode

        while not done:
            action = model.get_action(state)
            actions_taken.append(action)
            next_state, reward, done, _ = env.step(action)
            loss = model.update(optimizer, criterion, state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # Collect data for training visualization
        episode_data = {
            'episode': episode,
            'loss': loss, 
            'reward': total_reward,
            'actions': actions_taken 
        }
        training_data.append(episode_data)

        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}/{config.training.num_episodes}, Total Reward: {total_reward:.2f}, Loss: {loss:.4f}")
            visualization.plot_training_progress(training_data) # Visualize training progress

def evaluate_model(model: TradingModel, env: CryptoTradingEnv, config: DictConfig) -> float:
    total_reward = 0
    num_episodes = config.evaluation.num_episodes

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = model.get_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        total_reward += episode_reward

    avg_reward = total_reward / num_episodes
    logger.info(f"Evaluation - Average Reward: {avg_reward:.2f}")
    return avg_reward

def live_trading(model: TradingModel, data_provider: BinanceDataProvider, feature_engineer: FeatureEngineer, config: DictConfig, visualization: Visualization):
    portfolio_values = []
    returns = []
    order_history = []

    while True:
        # Fetch latest data
        latest_data = data_provider.get_data(config.exchange.symbol, config.exchange.timeframe, 
                                             limit=config.live_trading.lookback_period)
        
        # Preprocess data
        processed_data = preprocess_data(latest_data, feature_engineer)
        
        # Get the latest state
        latest_state = processed_data.iloc[-1].values
        
        # Get model prediction
        action = model.get_action(latest_state)
        
        # Execute trade based on the action
        if action == 1:  # Buy
            order = data_provider.place_order(config.exchange.symbol, 'market', 'buy', config.live_trading.trade_amount)
            logger.info(f"Buy order placed: {order}")
            # ... (Update portfolio_values, returns, order_history)
        elif action == 2:  # Sell
            order = data_provider.place_order(config.exchange.symbol, 'market', 'sell', config.live_trading.trade_amount)
            logger.info(f"Sell order placed: {order}")
            # ... (Update portfolio_values, returns, order_history)
        else:
            logger.info("Hold position")
        
        # Visualizations (Update periodically)
        visualization.plot_price_chart(latest_data)
        performance_data = {
            'portfolio_value': portfolio_values,
            'returns': returns,
            # ...
        }
        visualization.plot_performance_metrics(performance_data)
        visualization.display_order_history(order_history)

        # Wait for next trading interval
        time.sleep(config.live_trading.interval)

@hydra.main(config_path="../../config", config_name="base_config", version_base=None)
def main(cfg: DictConfig):
    setup_logging(cfg.logging)
    visualization = Visualization() # Instantiate for all modes
    
    # Initialize data provider
    data_provider = BinanceDataProvider(api_key=cfg.exchange.api_key, api_secret=cfg.exchange.api_secret)
    
    # Fetch historical data
    historical_data = data_provider.get_data(cfg.exchange.symbol, cfg.exchange.timeframe, 
                                             start_date=cfg.data.start_date, end_date=cfg.data.end_date)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Preprocess data
    processed_data = preprocess_data(historical_data, feature_engineer)
    
    # Create environment
    env = create_environment(processed_data, cfg)
    
    # Initialize model
    model = TradingModel(input_size=len(feature_engineer.get_feature_names()), 
                         hidden_size=cfg.model.hidden_size, 
                         num_layers=cfg.model.num_layers, 
                         output_size=env.action_space.n)
    
    if cfg.mode == "train":
        # Train the model
        train_model(model, env, cfg, visualization)
        
        # Save the trained model
        torch.save(model.state_dict(), cfg.paths.model_save_path)
        logger.info(f"Model saved to {cfg.paths.model_save_path}")
        
        # Evaluate the model
        evaluate_model(model, env, cfg)
    
    elif cfg.mode == "evaluate":
        # Load the trained model
        model.load_state_dict(torch.load(cfg.paths.model_save_path))
        logger.info(f"Model loaded from {cfg.paths.model_save_path}")
        
        # Evaluate the model
        evaluate_model(model, env, cfg)
    
    elif cfg.mode == "live":
        # Load the trained model
        model.load_state_dict(torch.load(cfg.paths.model_save_path))
        logger.info(f"Model loaded from {cfg.paths.model_save_path}")
        
        # Start live trading
        live_trading(model, data_provider, feature_engineer, cfg, visualization)
    
    else:
        logger.error(f"Invalid mode: {cfg.mode}")

if __name__ == "__main__":
    main()
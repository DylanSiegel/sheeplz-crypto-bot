# src/experiment_manager.py

import mlflow
import optuna
from src.agent_manager import AgentManager, TradingAgent
from environments.crypto_trading_env import CryptoTradingEnv
from models.model import TradingModel
from src.rewards import ProfitReward, SharpeRatioReward
from src.data_acquisition import BinanceDataProvider
from src.utils import get_logger
import torch

logger = get_logger()

class ExperimentManager:
    """
    Manages experiment tracking and hyperparameter optimization.
    """

    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def run_experiment(self, params: dict, trade_history: pd.DataFrame):
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Initialize components based on params
            data_provider = BinanceDataProvider(api_key=params['api_key'], api_secret=params['api_secret'])
            reward_function = ProfitReward() if params['reward_type'] == 'profit' else SharpeRatioReward()
            env = CryptoTradingEnv(data_provider=data_provider, reward_function=reward_function, initial_balance=params['initial_balance'])
            model = TradingModel(input_size=env.observation_space.shape[0],
                                 hidden_size=params['hidden_size'],
                                 output_size=env.action_space.n)
            model.load_state_dict(torch.load(params['model_path']))
            model.eval()

            # Initialize agents
            agents = [TradingAgent.remote(env, model, reward_function) for _ in range(params['num_agents'])]
            agent_manager = AgentManager(agents)

            # Train agents
            results = agent_manager.train_agents(num_episodes=params['num_episodes'])

            # Log metrics
            average_reward = sum(results) / len(results)
            mlflow.log_metric("average_reward", average_reward)

            logger.info(f"Experiment completed with average_reward: {average_reward}")

            return average_reward

    def optimize_hyperparameters(self, objective_function, num_trials: int):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_function, n_trials=num_trials)
        return study.best_params

# Example usage
# experiment_manager = ExperimentManager(experiment_name="Trading Experiment")
# best_params = experiment_manager.optimize_hyperparameters(objective_function, num_trials=50)

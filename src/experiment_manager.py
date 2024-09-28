# File: src/experiment_manager.py

import mlflow
import optuna
from src.agent_manager import AgentManager, TradingAgent
from environments.crypto_trading_env import CryptoTradingEnv
from models.model import TradingModel
from src.rewards import ProfitReward, SharpeRatioReward
from src.data_acquisition import BinanceDataProvider

class ExperimentManager:
    """
    Manages experiment tracking and hyperparameter optimization.
    """

    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def run_experiment(self, params: dict):
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Set up environment and agents
            data_provider = BinanceDataProvider(api_key=params['api_key'], api_secret=params['api_secret'])
            reward_function = ProfitReward() if params['reward_type'] == 'profit' else SharpeRatioReward()
            env = CryptoTradingEnv(data_provider, reward_function, initial_balance=params['initial_balance'])
            model = TradingModel(input_size=10, hidden_size=params['hidden_size'], output_size=3)
            
            agents = [TradingAgent.remote(env, model, reward_function) for _ in range(params['num_agents'])]
            agent_manager = AgentManager(agents)

            # Train agents
            results = agent_manager.train_agents(num_episodes=params['num_episodes'])

            # Calculate and log metrics
            avg_reward = sum(results) / len(results)
            mlflow.log_metric("average_reward", avg_reward)

            return avg_reward

    def optimize_hyperparameters(self, objective_function, num_trials: int):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_function, n_trials=num_trials)

        return study.best_params

# Example usage
# experiment_manager = ExperimentManager(experiment_name="Trading Experiment")

# def objective(trial):
#     params = {
#         'api_key': 'YOUR_API_KEY',
#         'api_secret': 'YOUR_API_SECRET',
#         'reward_type': trial.suggest_categorical('reward_type', ['profit', 'sharpe']),
#         'initial_balance': trial.suggest_float('initial_balance', 1000, 10000),
#         'hidden_size': trial.suggest_int('hidden_size', 32, 256),
#         'num_agents': trial.suggest_int('num_agents', 1, 8),
#         'num_episodes': 100
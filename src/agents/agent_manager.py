from typing import List, Dict, Any
import ray
from .trading_agent import TradingAgent
from environments.crypto_trading_env import CryptoTradingEnv
from src.models.lstm_model import TradingModel # Corrected import
from src.rewards.rewards import RewardFunction, get_reward_function
from utils.utils import get_logger

logger = get_logger(__name__)

@ray.remote
class RemoteTradingAgent(TradingAgent):
    pass

class AgentManager:
    def __init__(self, agent_configs: List[Dict[str, Any]], environments: Dict[str, CryptoTradingEnv]):
        ray.init(ignore_reinit_error=True)
        self.agents = []
        for config in agent_configs:
            env = environments[config['timeframe']]
            model = TradingModel(
                input_size=env.observation_space.shape[0],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=env.action_space.n
            )
            reward_function = get_reward_function(config['reward_function'])
            agent = RemoteTradingAgent.remote(env, model, reward_function, config['agent_id'])
            self.agents.append(agent)

    def train_agents(self, num_episodes: int):
        results = ray.get([agent.train.remote(num_episodes) for agent in self.agents])
        for result in results:
            logger.info(f"Agent {result['agent_id']} - Average Reward: {result['average_reward']:.2f}")
        return results

    def evaluate_agents(self, num_episodes: int):
        results = ray.get([agent.evaluate.remote(num_episodes) for agent in self.agents])
        for result in results:
            logger.info(f"Agent {result['agent_id']} - Evaluation Average Reward: {result['average_reward']:.2f}")
        return results

    def get_best_agent(self):
        evaluation_results = self.evaluate_agents(num_episodes=100)
        best_agent = max(evaluation_results, key=lambda x: x['average_reward'])
        return best_agent['agent_id']

    def save_agents(self, directory: str):
        ray.get([agent.save.remote(directory) for agent in self.agents])

    def load_agents(self, directory: str):
        ray.get([agent.load.remote(directory) for agent in self.agents])

    def shutdown(self):
        ray.shutdown()
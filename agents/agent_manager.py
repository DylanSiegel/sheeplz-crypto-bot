from typing import List, Dict, Any
from .trading_agent import TradingAgent
from environments.crypto_trading_env import CryptoTradingEnv
from models.trading_model import TradingModel
from src.rewards import RewardFunction, get_reward_function
from src.utils import get_logger

logger = get_logger(__name__)

class AgentManager:
    def __init__(self, agent_configs: List[Dict[str, Any]], environments: Dict[str, CryptoTradingEnv]):
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
            agent = TradingAgent(env, model, reward_function, config['agent_id'])
            self.agents.append(agent)

    def train_agents(self, num_episodes: int):
        results = []
        for agent in self.agents:
            avg_reward = agent.train(num_episodes)
            results.append({"agent_id": agent.agent_id, "average_reward": avg_reward})
            logger.info(f"Agent {agent.agent_id} - Average Reward: {avg_reward:.2f}")
        return results

    def evaluate_agents(self, num_episodes: int):
        results = []
        for agent in self.agents:
            avg_reward = agent.evaluate(num_episodes)
            results.append({"agent_id": agent.agent_id, "average_reward": avg_reward})
            logger.info(f"Agent {agent.agent_id} - Evaluation Average Reward: {avg_reward:.2f}")
        return results

    def get_best_agent(self):
        evaluation_results = self.evaluate_agents(num_episodes=100)
        best_agent = max(evaluation_results, key=lambda x: x['average_reward'])
        return best_agent['agent_id']

    def save_agents(self, directory: str):
        for agent in self.agents:
            agent.save(directory)

    def load_agents(self, directory: str):
        for agent in self.agents:
            agent.load(directory)
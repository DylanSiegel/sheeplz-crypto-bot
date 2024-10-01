# File: src/agents/agent_manager.py

from typing import List, Dict, Any
import ray
from src.agents.trading_agent import TradingAgent  # Corrected import
from src.environments.crypto_trading_env import CryptoTradingEnv  # Corrected import
from src.models.lstm_model import LSTMModel  # Corrected import
from src.rewards.rewards import RewardFunction, get_reward_function  # Corrected import
from src.utils.utils import get_logger  # Corrected import

logger = get_logger(__name__)

@ray.remote
class RemoteTradingAgent(TradingAgent):
    """
    Ray remote wrapper for TradingAgent.
    This allows TradingAgent instances to run in parallel as Ray actors.
    """
    pass

class AgentManager:
    """
    Manages multiple TradingAgent instances using Ray for parallel execution.
    Handles training, evaluation, saving, loading, and selection of the best-performing agents.
    """

    def __init__(self, agent_configs: List[Dict[str, Any]], environments: Dict[str, CryptoTradingEnv]):
        """
        Initializes the AgentManager with agent configurations and environments.

        Args:
            agent_configs (List[Dict[str, Any]]): List of agent configuration dictionaries.
                Each config should contain:
                    - 'timeframe': str, e.g., '1m'
                    - 'hidden_size': int, size of LSTM hidden layer
                    - 'num_layers': int, number of LSTM layers
                    - 'reward_function': str, e.g., 'profit', 'sharpe_ratio', 'combined'
                    - 'agent_id': str, unique identifier for the agent
            environments (Dict[str, CryptoTradingEnv]): Dictionary mapping timeframes to CryptoTradingEnv instances.
        """
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("Ray initialized for AgentManager.")

        self.agents = []
        self._initialize_agents(agent_configs, environments)

    def _initialize_agents(self, agent_configs: List[Dict[str, Any]], environments: Dict[str, CryptoTradingEnv]):
        """
        Initializes and launches remote trading agents based on configurations.

        Args:
            agent_configs (List[Dict[str, Any]]): List of agent configuration dictionaries.
            environments (Dict[str, CryptoTradingEnv]): Dictionary mapping timeframes to CryptoTradingEnv instances.
        """
        for config in agent_configs:
            timeframe = config.get('timeframe')
            if timeframe not in environments:
                logger.error(f"Timeframe '{timeframe}' not found in provided environments.")
                continue

            env = environments[timeframe]
            model = LSTMModel(
                input_size=env.observation_space.shape[0],
                hidden_size=config.get('hidden_size', 64),
                num_layers=config.get('num_layers', 2),
                output_size=env.action_space.n
            )
            reward_function = get_reward_function(config.get('reward_function', 'profit'))
            agent_id = config.get('agent_id', f"agent_{len(self.agents)+1}")

            # Instantiate the remote agent
            agent = RemoteTradingAgent.remote(env, model, reward_function, agent_id)
            self.agents.append(agent)
            logger.info(f"Initialized RemoteTradingAgent '{agent_id}' for timeframe '{timeframe}'.")

    def train_agents(self, num_episodes: int) -> List[Dict[str, Any]]:
        """
        Trains all agents for a specified number of episodes.

        Args:
            num_episodes (int): Number of training episodes.

        Returns:
            List[Dict[str, Any]]: List of training results from each agent.
        """
        if not self.agents:
            logger.warning("No agents to train.")
            return []

        logger.info(f"Starting training for {len(self.agents)} agents for {num_episodes} episodes each.")
        results = ray.get([agent.train.remote(num_episodes) for agent in self.agents])
        for result in results:
            agent_id = result.get('agent_id', 'Unknown')
            avg_reward = result.get('average_reward', 0.0)
            logger.info(f"Agent '{agent_id}' - Average Reward: {avg_reward:.2f}")
        return results

    def evaluate_agents(self, num_episodes: int) -> List[Dict[str, Any]]:
        """
        Evaluates all agents for a specified number of episodes.

        Args:
            num_episodes (int): Number of evaluation episodes.

        Returns:
            List[Dict[str, Any]]: List of evaluation results from each agent.
        """
        if not self.agents:
            logger.warning("No agents to evaluate.")
            return []

        logger.info(f"Starting evaluation for {len(self.agents)} agents for {num_episodes} episodes each.")
        results = ray.get([agent.evaluate.remote(num_episodes) for agent in self.agents])
        for result in results:
            agent_id = result.get('agent_id', 'Unknown')
            avg_reward = result.get('average_reward', 0.0)
            logger.info(f"Agent '{agent_id}' - Evaluation Average Reward: {avg_reward:.2f}")
        return results

    def get_best_agent(self) -> str:
        """
        Identifies the best-performing agent based on evaluation results.

        Returns:
            str: ID of the best-performing agent.
        """
        evaluation_results = self.evaluate_agents(num_episodes=100)
        if not evaluation_results:
            logger.warning("No evaluation results to determine the best agent.")
            return ""

        best_agent = max(evaluation_results, key=lambda x: x.get('average_reward', float('-inf')))
        best_agent_id = best_agent.get('agent_id', '')
        logger.info(f"Best agent identified: '{best_agent_id}' with Average Reward: {best_agent.get('average_reward', 0.0):.2f}")
        return best_agent_id

    def save_agents(self, directory: str):
        """
        Saves all agents' models and performance histories to the specified directory.

        Args:
            directory (str): Directory path to save the agents' data.
        """
        if not self.agents:
            logger.warning("No agents to save.")
            return

        logger.info(f"Saving all agents to directory: '{directory}'.")
        results = ray.get([agent.save.remote(directory) for agent in self.agents])
        logger.info("All agents have been saved successfully.")

    def load_agents(self, directory: str):
        """
        Loads all agents' models and performance histories from the specified directory.

        Args:
            directory (str): Directory path from where to load the agents' data.
        """
        if not self.agents:
            logger.warning("No agents to load.")
            return

        logger.info(f"Loading all agents from directory: '{directory}'.")
        results = ray.get([agent.load.remote(directory) for agent in self.agents])
        logger.info("All agents have been loaded successfully.")

    def shutdown(self):
        """
        Shuts down Ray to free up resources.
        """
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray has been shutdown successfully.")
        else:
            logger.info("Ray was not initialized; no need to shutdown.")

# File: src/agent_manager.py

from typing import List
import ray
from environments.crypto_trading_env import CryptoTradingEnv
from models.model import TradingModel
from src.rewards import RewardFunction
import torch

@ray.remote
class TradingAgent:
    """
    Represents a single trading agent.
    """

    def __init__(self, env: CryptoTradingEnv, model: TradingModel, reward_function: RewardFunction):
        self.env = env
        self.model = model
        self.reward_function = reward_function

    def train(self, num_episodes: int):
        total_reward = 0
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Get action from model
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    output = self.model(state_tensor)
                    action = torch.argmax(output, dim=1).item()

                # Take action in environment
                next_state, reward, done, _ = self.env.step(action)

                # Accumulate reward
                episode_reward += reward

                state = next_state

            total_reward += episode_reward

        average_reward = total_reward / num_episodes
        return average_reward

class AgentManager:
    """
    Manages multiple trading agents.
    """

    def __init__(self, agents: List[TradingAgent]):
        self.agents = agents

    def train_agents(self, num_episodes: int):
        # Train agents in parallel using Ray
        results = ray.get([agent.train.remote(num_episodes) for agent in self.agents])
        return results

# Example usage
# ray.init()
# env = CryptoTradingEnv(data_provider, reward_function)
# model = TradingModel(...)
# agent = TradingAgent.remote(env, model, reward_function)
# agent_manager = AgentManager([agent])
# results = agent_manager.train_agents(num_episodes=100)

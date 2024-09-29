import ray
from typing import List, Dict
import numpy as np
from environments.crypto_trading_env import CryptoTradingEnv
from models.model import TradingModel
from src.rewards import RewardFunction

@ray.remote
class TradingAgent:
    def __init__(self, env: CryptoTradingEnv, model: TradingModel, reward_function: RewardFunction, agent_id: str):
        self.env = env
        self.model = model
        self.reward_function = reward_function
        self.agent_id = agent_id
        self.performance_history = []

    def train(self, num_episodes: int):
        total_reward = 0
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.model.update(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

            self.performance_history.append(episode_reward)
            total_reward += episode_reward

        average_reward = total_reward / num_episodes
        return {"agent_id": self.agent_id, "average_reward": average_reward, "performance_history": self.performance_history}

class AgentManager:
    def __init__(self, agents: List[TradingAgent]):
        self.agents = agents

    def train_agents(self, num_episodes: int) -> List[Dict]:
        results = ray.get([agent.train.remote(num_episodes) for agent in self.agents])
        return results

    def get_best_agent(self) -> str:
        results = self.train_agents(num_episodes=100)  # Train for 100 episodes to determine the best agent
        best_agent = max(results, key=lambda x: x['average_reward'])
        return best_agent['agent_id']

    def ensemble_prediction(self, state) -> int:
        predictions = ray.get([agent.model.predict.remote(state) for agent in self.agents])
        return np.argmax(np.bincount(predictions))
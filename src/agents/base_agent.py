# File: src/agents/base_agent.py

from abc import ABC, abstractmethod
from src.environments.crypto_trading_env import CryptoTradingEnv
from src.models.base_model import BaseModel
from src.rewards.base_reward import RewardFunction

class BaseAgent(ABC):
    @abstractmethod
    def train(self, num_episodes: int):
        """
        Trains the agent over a specified number of episodes.
        """
        pass

    @abstractmethod
    def act(self, state):
        """
        Determines the action to take based on the current state.
        """
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int):
        """
        Evaluates the agent's performance over a specified number of episodes.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Saves the agent's model and performance history.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Loads the agent's model and performance history.
        """
        pass

# File: tests/test_agent_manager.py

import unittest
from unittest.mock import MagicMock
from src.agent_manager import TradingAgent, AgentManager
from environments.crypto_trading_env import CryptoTradingEnv
from models.model import TradingModel
from src.rewards import ProfitReward

class TestAgentManager(unittest.TestCase):

    def setUp(self):
        # Mock environment, model, and reward function
        self.mock_env = MagicMock(spec=CryptoTradingEnv)
        self.mock_model = MagicMock(spec=TradingModel)
        self.mock_reward_function = MagicMock(spec=ProfitReward)

        # Create a TradingAgent instance
        self.agent = TradingAgent(self.mock_env, self.mock_model, self.mock_reward_function)

        # Initialize AgentManager with a list of agents
        self.agent_manager = AgentManager([self.agent])

    def test_train_agents(self):
        # Setup mock behavior
        self.mock_env.reset.return_value = [1,2,3]
        self.mock_env.step.return_value = ([4,5,6], 1.0, True, {})
        self.mock_model.return_value = MagicMock(argmax=MagicMock(return_value=1))

        # Execute training
        results = self.agent_manager.train_agents(num_episodes=1)

        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 1.0)  # Assuming total_reward per episode is 1.0

if __name__ == '__main__':
    unittest.main()

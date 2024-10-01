# File: src/agents/trading_agent.py

import torch
import torch.nn as nn
from src.environments.crypto_trading_env import CryptoTradingEnv
from src.models.lstm_model import LSTMModel  # Corrected import
from src.rewards.rewards import RewardFunction
import logging

logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(self, env: CryptoTradingEnv, model: LSTMModel, reward_function: RewardFunction, agent_id: str, learning_rate: float = 0.001):
        """
        Initializes the TradingAgent.

        Args:
            env (CryptoTradingEnv): The trading environment.
            model (LSTMModel): The DRL model.
            reward_function (RewardFunction): The reward function.
            agent_id (str): Unique identifier for the agent.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.env = env
        self.model = model
        self.reward_function = reward_function
        self.agent_id = agent_id
        self.performance_history = []

        # Initialize optimizer and criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, num_episodes: int):
        """
        Trains the agent over a specified number of episodes.

        Args:
            num_episodes (int): Number of training episodes.

        Returns:
            float: Average reward over episodes.
        """
        total_reward = 0
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            done = False
            episode_reward = 0
            loss = 0.0  # Initialize loss for the episode

            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)

                # Extract necessary components from state and next_state for reward calculation
                current_price = state[-2]  # Assuming second last element is current_price
                next_price = next_state[-2]  # Assuming second last element is next_price
                portfolio_value = next_state[-1]  # Assuming last element is portfolio_value

                adjusted_reward = self.reward_function.calculate_reward(
                    action, current_price, next_price, portfolio_value
                )

                # Update the model
                loss = self.model.update(
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    state=state,
                    action=action,
                    reward=adjusted_reward,
                    next_state=next_state,
                    done=done
                )

                episode_reward += adjusted_reward
                state = next_state

            self.performance_history.append({
                'episode': episode,
                'reward': episode_reward,
                'loss': loss
            })
            total_reward += episode_reward

            if episode % 10 == 0 or episode == 1:
                logger.info(f"Agent {self.agent_id} - Episode {episode}/{num_episodes}, "
                            f"Reward: {episode_reward:.2f}, Loss: {loss:.4f}")

        average_reward = total_reward / num_episodes
        logger.info(f"Agent {self.agent_id} - Training completed. Average Reward: {average_reward:.2f}")
        return average_reward

    def act(self, state):
        """
        Determines the action to take based on the current state.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Action to take.
        """
        return self.model.get_action(state)

    def evaluate(self, num_episodes: int):
        """
        Evaluates the agent over a specified number of episodes.

        Args:
            num_episodes (int): Number of evaluation episodes.

        Returns:
            float: Average reward over episodes.
        """
        total_reward = 0
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.act(state)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

            total_reward += episode_reward
            logger.info(f"Agent {self.agent_id} - Evaluation Episode {episode} - Reward: {episode_reward:.2f}")

        average_reward = total_reward / num_episodes
        logger.info(f"Agent {self.agent_id} - Evaluation completed. Average Reward: {average_reward:.2f}")
        return average_reward

    def save(self, path: str):
        """
        Saves the agent's model and performance history.

        Args:
            path (str): Directory path to save the agent's data.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'performance_history': self.performance_history
        }, f"{path}/{self.agent_id}.pth")
        logger.info(f"Agent {self.agent_id} - Model and performance history saved to '{path}/{self.agent_id}.pth'.")

    def load(self, path: str):
        """
        Loads the agent's model and performance history.

        Args:
            path (str): Directory path from where to load the agent's data.
        """
        checkpoint = torch.load(f"{path}/{self.agent_id}.pth", map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.performance_history = checkpoint.get('performance_history', [])
        logger.info(f"Agent {self.agent_id} - Model and performance history loaded from '{path}/{self.agent_id}.pth'.")

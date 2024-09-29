import torch
from environments.crypto_trading_env import CryptoTradingEnv
from models.trading_model import TradingModel
from src.rewards import RewardFunction

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
                adjusted_reward = self.reward_function.calculate_reward(state, action, reward, next_state, done)
                self.model.update(state, action, adjusted_reward, next_state, done)
                episode_reward += adjusted_reward
                state = next_state

            self.performance_history.append(episode_reward)
            total_reward += episode_reward

        return total_reward / num_episodes

    def act(self, state):
        return self.model.get_action(state)

    def evaluate(self, num_episodes: int):
        total_reward = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
        return total_reward / num_episodes

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'performance_history': self.performance_history
        }, f"{path}/{self.agent_id}.pth")

    def load(self, path: str):
        checkpoint = torch.load(f"{path}/{self.agent_id}.pth")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.performance_history = checkpoint['performance_history']
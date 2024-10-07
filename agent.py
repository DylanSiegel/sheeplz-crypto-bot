# File: agent.py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorCritic, self).__init__()
        # Common feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        )
        # Actor network
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        # Critic network
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        feature = self.feature(state)
        mean = self.actor_mean(feature)
        std = self.actor_log_std.exp()
        value = self.critic(feature)
        return mean, std, value

class PPOAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, eps_clip=0.2, c1=0.5, c2=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.c1 = c1  # Critic loss coefficient
        self.c2 = c2  # Entropy coefficient

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mean, std, _ = self.model(state)
        dist = Normal(mean, std)
        action = dist.rsample()
        tanh_action = torch.tanh(action)
        log_prob = dist.log_prob(action) - torch.log(1 - tanh_action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1)
        action_np = tanh_action.cpu().numpy()
        return action_np, log_prob.item()

    def compute_returns(self, rewards, masks, values, next_value):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def update(self, trajectories):
        states = torch.FloatTensor(np.array(trajectories['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(trajectories['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(trajectories['log_probs'])).to(self.device)
        returns = torch.FloatTensor(np.array(trajectories['returns'])).to(self.device)
        values = torch.FloatTensor(np.array(trajectories['values'])).to(self.device)
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(4):  # Number of epochs
            mean, std, current_values = self.model(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions)
            tanh_actions = torch.tanh(actions)
            new_log_probs -= torch.log(1 - tanh_actions.pow(2) + 1e-7)
            new_log_probs = new_log_probs.sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = (new_log_probs - old_log_probs).exp()

            # Surrogate function
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.c1 * (returns - current_values.squeeze()).pow(2).mean()
            loss = actor_loss + critic_loss - self.c2 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

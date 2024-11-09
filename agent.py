# File: agent.py

import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        lr=1e-4,
        buffer_size=100000,
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)

        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.steps_done = 0

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state):
        self.steps_done += 1
        epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        transitions = random.sample(self.buffer, self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
            expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)

        loss = nn.functional.mse_loss(state_action_values.squeeze(), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

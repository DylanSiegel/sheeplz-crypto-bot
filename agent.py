import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
from torch.utils.tensorboard import SummaryWriter
import logging

from config import MetaSACConfig
from networks import MetaSACActor, MetaSACCritic, MetaController
from replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

class MetaSACAgent(nn.Module):
    """Meta Soft Actor-Critic agent."""
    def __init__(self, config: MetaSACConfig):
        super().__init__()
        self.config = config
        self.device = config.device

        # Initialize networks
        self.actor = MetaSACActor(config).to(self.device)
        self.critic1 = MetaSACCritic(config).to(self.device)
        self.critic2 = MetaSACCritic(config).to(self.device)
        self.critic_target1 = MetaSACCritic(config).to(self.device)
        self.critic_target2 = MetaSACCritic(config).to(self.device)

        # Copy weights
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # Initialize meta-controller
        self.meta_controller = MetaController(config).to(self.device)

        # Setup optimizers
        self.setup_optimizers()

        # Initialize alpha parameter
        self.alpha = nn.Parameter(torch.tensor(config.alpha, dtype=torch.float32, device=self.device))
        self.target_entropy = -torch.prod(torch.tensor([config.action_dim], dtype=torch.float32)).to(self.device)

        # Setup tensorboard writer
        self.writer = SummaryWriter()
        self.train_steps = 0
        self.reward_history = []

    def setup_optimizers(self) -> None:
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.lr)
        self.meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=self.config.meta_lr)

        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='min', factor=0.5, patience=5
        )
        self.critic1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic1_optimizer, mode='min', factor=0.5, patience=5
        )
        self.critic2_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic2_optimizer, mode='min', factor=0.5, patience=5
        )

    def select_action(self, state: np.ndarray, time_step: int, eval: bool = False) -> torch.Tensor:
        if state.shape[-1] != self.config.state_dim:
            raise ValueError(f"Expected state dimension {self.config.state_dim}, got {state.shape[-1]}")

        if np.isnan(state).any():
            logger.warning("NaN detected in state input, replacing with zeros")
            state = np.nan_to_num(state)

        self.actor.eval() if eval else self.actor.train()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            time_tensor = torch.tensor(time_step, dtype=torch.float32).unsqueeze(0).to(self.device)

            mu, log_sigma = self.actor(state_tensor, time_tensor)

            if eval:
                action = mu
            else:
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                action = torch.tanh(dist.rsample())

            action = torch.clamp(action, -1.0, 1.0)
            return action

    def compute_q_targets(self, rewards: torch.Tensor, next_states: torch.Tensor,
                          time_steps: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_actions = []
            next_log_probs = []

            for ns, ts in zip(next_states, time_steps):
                action = self.select_action(ns.cpu().numpy(), ts.item(), eval=False).unsqueeze(0)
                next_actions.append(action)

                mu, log_sigma = self.actor(ns.unsqueeze(0), ts.unsqueeze(0))
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                z = dist.rsample()
                action_tensor = torch.tanh(z)
                log_prob = dist.log_prob(z) - torch.log(1 - action_tensor.pow(2) + self.config.epsilon)
                next_log_probs.append(log_prob.sum(-1))

            next_actions = torch.cat(next_actions, dim=0).to(self.device)
            next_log_probs = torch.cat(next_log_probs).unsqueeze(-1).to(self.device)

            q_target1 = self.critic_target1(next_states, next_actions, time_steps)
            q_target2 = self.critic_target2(next_states, next_actions, time_steps)
            q_target = torch.min(q_target1, q_target2)

            return rewards + (1.0 - dones) * self.config.gamma * (q_target - self.alpha * next_log_probs)

    def update_critics(self, states: torch.Tensor, actions: torch.Tensor,
                       time_steps: torch.Tensor, q_targets: torch.Tensor) -> Tuple[float, float]:
        q_value1 = self.critic1(states, actions, time_steps)
        q_value2 = self.critic2(states, actions, time_steps)

        critic1_loss = F.mse_loss(q_value1, q_targets)
        critic2_loss = F.mse_loss(q_value2, q_targets)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.max_grad_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.max_grad_norm)
        self.critic2_optimizer.step()

        return critic1_loss.item(), critic2_loss.item()

    def update_actor(self, states: torch.Tensor, time_steps: torch.Tensor) -> float:
        mu, log_sigma = self.actor(states, time_steps)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        actions = torch.tanh(z)
        log_probs = dist.log_prob(z) - torch.log(1 - actions.pow(2) + self.config.epsilon)
        log_probs = log_probs.sum(-1, keepdim=True)

        q_values = torch.min(
            self.critic1(states, actions, time_steps),
            self.critic2(states, actions, time_steps)
        )

        actor_loss = (self.alpha * log_probs - q_values).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        return actor_loss.item()

    def update_meta_controller(self, meta_input: torch.Tensor, log_probs: torch.Tensor, rewards: torch.Tensor) -> float:
        mean = torch.mean(rewards, dim=0, keepdim=True)
        variance = torch.var(rewards, dim=0, keepdim=True)
        reward_stats = torch.cat([mean, variance], dim=-1).to(self.device)

        batch_size = meta_input.size(0)
        reward_stats = reward_stats.repeat(batch_size, 1)

        meta_output = self.meta_controller(meta_input, reward_stats)
        learning_rate_actor, learning_rate_critic, learning_rate_alpha, tau, gamma = meta_output
        alpha_target_loss = -(learning_rate_alpha - torch.log(self.alpha)) * (log_probs + self.target_entropy).detach().mean()

        self.meta_optimizer.zero_grad()
        alpha_target_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.meta_controller.parameters(), self.config.max_grad_norm)
        self.meta_optimizer.step()

        return alpha_target_loss.item()

    def soft_update(self, target_network: nn.Module, source_network: nn.Module) -> None:
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )

    def update_params(self, replay_buffer: ReplayBuffer, meta_input: np.ndarray,
                      time_memory: List[int]) -> Dict[str, float]:
        if len(replay_buffer) < self.config.batch_size:
            return {}

        try:
            batch, indices, weights = replay_buffer.sample(self.config.batch_size)
        except ValueError as e:
            logger.error(f"Failed to sample from replay buffer: {str(e)}")
            return {}

        states = torch.FloatTensor(np.stack([item[0] for item in batch])).to(self.device)
        actions = torch.FloatTensor(np.stack([item[1] for item in batch])).to(self.device)
        rewards = torch.FloatTensor(np.stack([item[2] for item in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack([item[3] for item in batch])).to(self.device)
        dones = torch.FloatTensor(np.stack([item[4] for item in batch])).unsqueeze(1).to(self.device)
        meta_input_tensor = torch.FloatTensor(meta_input).to(self.device)
        time_steps = torch.tensor(time_memory, dtype=torch.float32).to(self.device)

        q_targets = self.compute_q_targets(rewards, next_states, time_steps, dones)

        critic1_loss, critic2_loss = self.update_critics(states, actions, time_steps, q_targets)

        actor_loss = self.update_actor(states, time_steps)

        mu, log_sigma = self.actor(states, time_steps)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        actions_sampled = torch.tanh(z)
        log_probs = dist.log_prob(z) - torch.log(1 - actions_sampled.pow(2) + self.config.epsilon)
        log_probs = log_probs.sum(-1, keepdim=True)
        meta_loss = self.update_meta_controller(meta_input_tensor, log_probs, rewards)

        self.soft_update(self.critic_target1, self.critic1)
        self.soft_update(self.critic_target2, self.critic2)

        self.actor_scheduler.step(actor_loss)
        self.critic1_scheduler.step(critic1_loss)
        self.critic2_scheduler.step(critic2_loss)

        self.train_steps += 1
        self.writer.add_scalar('Loss/actor', actor_loss, self.train_steps)
        self.writer.add_scalar('Loss/critic1', critic1_loss, self.train_steps)
        self.writer.add_scalar('Loss/critic2', critic2_loss, self.train_steps)
        self.writer.add_scalar('Loss/meta', meta_loss, self.train_steps)
        self.writer.add_scalar('Parameters/alpha', self.alpha.item(), self.train_steps)

        return {
            'actor_loss': actor_loss,
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
            'meta_loss': meta_loss,
            'alpha': self.alpha.item()
        }

    def save(self, path: str) -> None:
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'critic_target1_state_dict': self.critic_target1.state_dict(),
                'critic_target2_state_dict': self.critic_target2.state_dict(),
                'meta_controller_state_dict': self.meta_controller.state_dict(),
                'alpha': self.alpha.detach().cpu().numpy(),
                'config': self.config,
                'train_steps': self.train_steps
            }, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.critic_target1.load_state_dict(checkpoint['critic_target1_state_dict'])
            self.critic_target2.load_state_dict(checkpoint['critic_target2_state_dict'])
            self.meta_controller.load_state_dict(checkpoint['meta_controller_state_dict'])
            self.alpha.data.copy_(torch.tensor(checkpoint['alpha'], dtype=torch.float32, device=self.device))
            self.train_steps = checkpoint['train_steps']
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

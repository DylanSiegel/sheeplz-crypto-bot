# agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
from torch.utils.tensorboard import SummaryWriter
import logging
from torch.optim import RAdam, Lookahead

# Use the consolidated config from env/config.py
from env.config import EnvironmentConfig
from networks import (
    MetaSACActor, 
    MetaSACCritic, 
    MetaController, 
    PolicyDistiller, 
    MarketModeClassifier, 
    HighLevelPolicy
)
from replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

class MetaSACAgent(nn.Module):
    """Meta Soft Actor-Critic agent."""
    def __init__(self, config: EnvironmentConfig):
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

        # Initialize policy distiller and specialists (example only)
        self.specialist_policies = [MetaSACActor(config).to(self.device) for _ in range(3)]
        self.policy_distiller = PolicyDistiller(self.specialist_policies).to(self.device)

        # Initialize market mode classifier
        self.market_mode_classifier = MarketModeClassifier(config.state_dim, config.hidden_dim).to(self.device)

        # Initialize high level policy
        self.high_level_policy = HighLevelPolicy(config.state_dim, config.hidden_dim).to(self.device)

        # Setup optimizers
        self.setup_optimizers()

        # Initialize alpha parameter (entropy coef)
        self.alpha = nn.Parameter(torch.tensor(config.alpha, dtype=torch.float32, device=self.device))
        self.target_entropy = -torch.prod(torch.tensor([config.action_dim], dtype=torch.float32)).to(self.device)

        # Setup tensorboard writer
        self.writer = SummaryWriter()
        self.train_steps = 0
        self.reward_history = []

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_capacity)

    def setup_optimizers(self) -> None:
        self.actor_optimizer = Lookahead(RAdam(self.actor.parameters(), lr=self.config.lr))
        self.critic1_optimizer = Lookahead(RAdam(self.critic1.parameters(), lr=self.config.lr))
        self.critic2_optimizer = Lookahead(RAdam(self.critic2.parameters(), lr=self.config.lr))
        self.meta_optimizer = Lookahead(RAdam(self.meta_controller.parameters(), lr=self.config.meta_lr))
        self.distiller_optimizer = Lookahead(RAdam(self.policy_distiller.parameters(), lr=self.config.lr))
        self.market_mode_optimizer = Lookahead(RAdam(self.market_mode_classifier.parameters(), lr=self.config.lr))
        self.high_level_optimizer = Lookahead(RAdam(self.high_level_policy.parameters(), lr=self.config.lr))

        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode="min", factor=0.5, patience=5
        )
        self.critic1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic1_optimizer, mode="min", factor=0.5, patience=5
        )
        self.critic2_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic2_optimizer, mode="min", factor=0.5, patience=5
        )
        self.high_level_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.high_level_optimizer, mode="min", factor=0.5, patience=5
        )
        self.meta_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.meta_optimizer, mode="min", factor=0.5, patience=5
        )
        self.distiller_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.distiller_optimizer, mode="min", factor=0.5, patience=5
        )

    def select_action(self, state: np.ndarray, time_step: int, eval: bool = False) -> torch.Tensor:
        """
        Returns an action in [-1,1] for each dimension. The second dimension
        can be mapped to leverage in the environment from [0..1] -> [1..max_leverage].
        """
        if state.shape[-1] != self.config.state_dim:
            raise ValueError(f"Expected state dimension {self.config.state_dim}, got {state.shape[-1]}")

        if np.isnan(state).any():
            logger.warning("NaN detected in state input, replacing with zeros")
            state = np.nan_to_num(state)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        time_tensor = torch.tensor(time_step, dtype=torch.float32).unsqueeze(0).to(self.device)

        if not eval:
            # For training, we can do distillation approach
            with torch.no_grad():
                mu, log_sigma = self.policy_distiller(state_tensor, time_tensor)
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                action = torch.tanh(dist.rsample())
        else:
            self.actor.eval()
            with torch.no_grad():
                mu, _ = self.actor(state_tensor, time_tensor)
                action = mu

        action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0)

    def compute_q_targets(
        self, rewards: torch.Tensor, next_states: torch.Tensor, time_steps: torch.Tensor, dones: torch.Tensor,
        r_scaling: torch.Tensor, market_mode_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard SAC target with reward scaling from meta-controller.
        """
        with torch.no_grad():
            mu, log_sigma = self.actor(next_states, time_steps)
            sigma = torch.exp(log_sigma)
            dist = torch.distributions.Normal(mu, sigma)
            next_actions = torch.tanh(dist.rsample())
            log_probs = dist.log_prob(next_actions) - torch.log(1 - next_actions.pow(2) + self.config.epsilon)
            next_log_probs = log_probs.sum(-1, keepdim=True)

            q_target1 = self.critic_target1(next_states, next_actions, time_steps)
            q_target2 = self.critic_target2(next_states, next_actions, time_steps)
            q_target = torch.min(q_target1, q_target2)

            # Example: r_scaling is shape [batch_size, 3], same for market_mode_probs
            # This demonstration code is naive. Adjust if your logic differs.
            scaled_rewards = (
                r_scaling[:, 0] * rewards * market_mode_probs[:, 0].unsqueeze(1)
                + r_scaling[:, 1] * rewards * market_mode_probs[:, 1].unsqueeze(1)
                + r_scaling[:, 2] * rewards * market_mode_probs[:, 2].unsqueeze(1)
            )
            return scaled_rewards + (1.0 - dones) * self.config.gamma * (q_target - self.alpha * next_log_probs)

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

    def update_distiller(self, states: torch.Tensor, time_steps: torch.Tensor) -> float:
        """
        Similar objective as update_actor, but using the policy distiller.
        """
        mu_distilled, log_sigma_distilled = self.policy_distiller(states, time_steps)
        sigma_distilled = torch.exp(log_sigma_distilled)
        dist_distilled = torch.distributions.Normal(mu_distilled, sigma_distilled)
        z_distilled = dist_distilled.rsample()
        action_distilled = torch.tanh(z_distilled)
        log_prob_distilled = dist_distilled.log_prob(z_distilled) - torch.log(
            1 - action_distilled.pow(2) + self.config.epsilon
        )
        log_prob_distilled = log_prob_distilled.sum(-1, keepdim=True)

        q_values_distilled = torch.min(
            self.critic1(states, action_distilled, time_steps),
            self.critic2(states, action_distilled, time_steps)
        )
        distiller_loss = (self.alpha * log_prob_distilled - q_values_distilled).mean()

        self.distiller_optimizer.zero_grad()
        distiller_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_distiller.parameters(), self.config.max_grad_norm)
        self.distiller_optimizer.step()

        return distiller_loss.item()

    def update_market_mode_classifier(self, states: torch.Tensor, market_modes: torch.Tensor) -> float:
        mode_probs = self.market_mode_classifier(states)
        loss = F.cross_entropy(mode_probs, market_modes)
        self.market_mode_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.market_mode_classifier.parameters(), self.config.max_grad_norm)
        self.market_mode_optimizer.step()
        return loss.item()

    def update_meta_controller(self, meta_input: torch.Tensor, log_probs: torch.Tensor,
                               rewards: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Example of meta-controller that adjusts alpha, tau, gamma, reward scalings, etc.
        """
        mean = torch.mean(rewards, dim=0, keepdim=True)
        variance = torch.var(rewards, dim=0, keepdim=True)
        reward_stats = torch.cat([mean, variance], dim=-1).to(self.device)

        batch_size = meta_input.size(0)
        reward_stats = reward_stats.repeat(batch_size, 1)

        meta_output = self.meta_controller(meta_input, reward_stats)
        # Unpack 8 outputs
        (
            learning_rate_actor, learning_rate_critic, learning_rate_alpha,
            tau, gamma, r_scaling_momentum, r_scaling_reversal, r_scaling_volatility
        ) = meta_output

        # Simple alpha target loss
        alpha_target_loss = -(learning_rate_alpha - torch.log(self.alpha)) * (log_probs + self.target_entropy).detach().mean()

        self.meta_optimizer.zero_grad()
        alpha_target_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_controller.parameters(), self.config.max_grad_norm)
        self.meta_optimizer.step()

        # Optionally, update actual hyperparameters in code if desired:
        # self.alpha.data = ...

        return alpha_target_loss.item(), torch.stack(
            [r_scaling_momentum, r_scaling_reversal, r_scaling_volatility], dim=-1
        )

    def update_high_level_policy(self, states: torch.Tensor, advantages: torch.Tensor) -> float:
        probs = self.high_level_policy(states)
        log_probs = torch.log(probs + 1e-10)
        policy_loss = -(log_probs * advantages).mean()

        self.high_level_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level_policy.parameters(), self.config.max_grad_norm)
        self.high_level_optimizer.step()
        return policy_loss.item()

    def soft_update(self, target_network: nn.Module, source_network: nn.Module) -> None:
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )

    def update_params(
        self,
        replay_buffer: ReplayBuffer,
        meta_input: np.ndarray,
        time_memory: List[int],
        update_steps: int = 1
    ) -> Dict[str, float]:
        """
        Possibly do multiple gradient steps each time this is called (update_steps param).
        """
        final_info = {}
        for _ in range(update_steps):
            if len(replay_buffer) < self.config.batch_size:
                return final_info

            batch, indices, weights = replay_buffer.sample(self.config.batch_size)

            states = torch.FloatTensor(np.stack([item[0] for item in batch])).to(self.device)
            actions = torch.FloatTensor(np.stack([item[1] for item in batch])).to(self.device)
            rewards = torch.FloatTensor(np.stack([item[2] for item in batch])).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.stack([item[3] for item in batch])).to(self.device)
            dones = torch.FloatTensor(np.stack([item[4] for item in batch])).unsqueeze(1).to(self.device)
            time_steps = torch.tensor(np.stack([item[5] for item in batch]), dtype=torch.float32).to(self.device)

            meta_input_tensor = torch.FloatTensor(meta_input).to(self.device)
            # Clear meta inputs if you want them single-use
            # meta_input[:] = []

            # Market mode classification
            # Just random mode for demonstration
            market_modes = torch.randint(0, self.config.num_market_modes, (states.shape[0],)).to(self.device)
            market_mode_loss = self.update_market_mode_classifier(states, market_modes)

            # Collect log_probs for meta
            with torch.no_grad():
                mu, log_sigma = self.actor(states, time_steps)
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                z = dist.rsample()
                actions_ = torch.tanh(z)
                log_probs = dist.log_prob(z) - torch.log(1 - actions_.pow(2) + self.config.epsilon)
                log_probs = log_probs.sum(-1, keepdim=True)

            meta_loss_val, r_scaling = self.update_meta_controller(meta_input_tensor, log_probs, rewards)

            # Get market mode probabilities
            market_mode_probs = self.market_mode_classifier(states)

            q_targets = self.compute_q_targets(
                rewards, next_states, time_steps, dones, r_scaling, market_mode_probs
            )

            critic1_loss, critic2_loss = self.update_critics(states, actions, time_steps, q_targets)
            actor_loss_val = self.update_actor(states, time_steps)
            distiller_loss_val = self.update_distiller(states, time_steps)

            # High-level policy example
            advantages = q_targets - torch.mean(q_targets, dim=0, keepdim=True)
            high_level_loss_val = self.update_high_level_policy(states, advantages)

            # Soft update
            self.soft_update(self.critic_target1, self.critic1)
            self.soft_update(self.critic_target2, self.critic2)

            # Optionally update your LR schedulers
            self.actor_scheduler.step(actor_loss_val)
            self.critic1_scheduler.step(critic1_loss)
            self.critic2_scheduler.step(critic2_loss)
            self.high_level_scheduler.step(high_level_loss_val)
            self.meta_scheduler.step(meta_loss_val)
            self.distiller_scheduler.step(distiller_loss_val)

            # Update priorities if needed
            # e.g., new_priorities = ...
            # replay_buffer.update_priorities(indices, new_priorities)

            self.train_steps += 1
            self.writer.add_scalar("Loss/actor", actor_loss_val, self.train_steps)
            self.writer.add_scalar("Loss/critic1", critic1_loss, self.train_steps)
            self.writer.add_scalar("Loss/critic2", critic2_loss, self.train_steps)
            self.writer.add_scalar("Loss/meta", meta_loss_val, self.train_steps)
            self.writer.add_scalar("Loss/distiller", distiller_loss_val, self.train_steps)
            self.writer.add_scalar("Loss/market_mode", market_mode_loss, self.train_steps)
            self.writer.add_scalar("Loss/high_level_policy", high_level_loss_val, self.train_steps)
            self.writer.add_scalar("Parameters/alpha", self.alpha.item(), self.train_steps)

            final_info = {
                "actor_loss": actor_loss_val,
                "critic1_loss": critic1_loss,
                "critic2_loss": critic2_loss,
                "meta_loss": meta_loss_val,
                "distiller_loss": distiller_loss_val,
                "market_mode_loss": market_mode_loss,
                "high_level_loss": high_level_loss_val,
                "alpha": self.alpha.item(),
            }

        return final_info

    def save(self, path: str) -> None:
        config_essentials = {
            "action_dim": self.config.action_dim,
            "state_dim": self.config.state_dim,
            "hidden_dim": self.config.hidden_dim,
            "time_encoding_dim": self.config.time_encoding_dim,
            "num_market_modes": self.config.num_market_modes,
        }
        try:
            torch.save(
                {
                    "actor_state_dict": self.actor.state_dict(),
                    "critic1_state_dict": self.critic1.state_dict(),
                    "critic2_state_dict": self.critic2.state_dict(),
                    "critic_target1_state_dict": self.critic_target1.state_dict(),
                    "critic_target2_state_dict": self.critic_target2.state_dict(),
                    "meta_controller_state_dict": self.meta_controller.state_dict(),
                    "distiller_state_dict": self.policy_distiller.state_dict(),
                    "market_mode_classifier_state_dict": self.market_mode_classifier.state_dict(),
                    "high_level_policy_state_dict": self.high_level_policy.state_dict(),
                    "alpha": self.alpha.detach().cpu().numpy(),
                    "config": config_essentials,
                    "train_steps": self.train_steps,
                },
                path,
            )
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
            self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
            self.critic_target1.load_state_dict(checkpoint["critic_target1_state_dict"])
            self.critic_target2.load_state_dict(checkpoint["critic_target2_state_dict"])
            self.meta_controller.load_state_dict(checkpoint["meta_controller_state_dict"])
            self.policy_distiller.load_state_dict(checkpoint["distiller_state_dict"])
            self.market_mode_classifier.load_state_dict(checkpoint["market_mode_classifier_state_dict"])
            self.high_level_policy.load_state_dict(checkpoint["high_level_policy_state_dict"])
            self.alpha.data.copy_(torch.tensor(checkpoint["alpha"], dtype=torch.float32, device=self.device))
            self.train_steps = checkpoint["train_steps"]
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

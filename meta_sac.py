import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import unittest
import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class MetaSACConfig:
    """Configuration for MetaSAC agent and networks."""
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    attention_dim: int = 32
    meta_input_dim: int = 5
    time_encoding_dim: int = 10
    num_mlp_layers: int = 3
    dropout_rate: float = 0.1
    lr: float = 1e-3
    meta_lr: float = 1e-4
    alpha: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64 # Changed to 64 to be consistent with test
    max_grad_norm: float = 1.0
    epsilon: float = 1e-10
    device: torch.device = device
    replay_buffer_capacity: int = 1000000


# ------------------------- Helper Classes -------------------------
class APELU(nn.Module):
    """Advanced Parametric Exponential Linear Unit activation."""
    def __init__(self, alpha_init: float = 0.01, beta_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.alpha * x * torch.exp(self.beta * x))


# ------------------------- Modern MLP -------------------------
class ModernMLP(nn.Module):
    """Modern Multi-Layer Perceptron with layer normalization and dropout."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(in_features, out_features))
            if i != num_layers - 1:
                self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.activation = APELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        return x
        
# ------------------------- Time Encoding -------------------------
class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal positional encoding for time."""
    def __init__(self, time_encoding_dim: int):
        super().__init__()
        self.time_encoding_dim = time_encoding_dim
        self.frequencies = 10 ** (torch.arange(0, time_encoding_dim // 2) * (-2 / (time_encoding_dim // 2)))

    def forward(self, time_step: torch.Tensor) -> torch.Tensor:
        time_step = time_step.float()  # Cast to float
        scaled_time = time_step.unsqueeze(-1) * self.frequencies.to(time_step.device)
        sin_encodings = torch.sin(scaled_time)
        cos_encodings = torch.cos(scaled_time)

        if self.time_encoding_dim % 2 == 0:
          encoding = torch.cat([sin_encodings, cos_encodings], dim=-1)
        else:
          encoding = torch.cat([sin_encodings, cos_encodings, torch.zeros_like(cos_encodings[:, :1])], dim=-1)

        return encoding

# ------------------------- Adaptive Modulation MLP -------------------------
class TimeAwareBias(nn.Module):
    """Time-aware bias module for temporal adaptation."""
    def __init__(self, input_dim: int, time_encoding_dim: int = 10, hidden_dim: int = 20):
        super().__init__()
        self.time_embedding = nn.Linear(time_encoding_dim, hidden_dim)
        self.time_projection = nn.Linear(hidden_dim, input_dim)
        self.activation = APELU()

    def forward(self, time_encoding: torch.Tensor) -> torch.Tensor:
        x = self.time_embedding(time_encoding)
        x = self.activation(x)
        return self.time_projection(x)


class AdaptiveModulationMLP(nn.Module):
    """MLP with adaptive modulation based on time encoding."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout_rate: float = 0.1, time_encoding_dim: int = 10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.modulations = nn.ParameterList()
        self.time_biases = nn.ModuleList()
        self.sinusoidal_encoding = SinusoidalTimeEncoding(time_encoding_dim)
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(in_features, out_features))
            if i != num_layers - 1:
                self.norms.append(nn.LayerNorm(hidden_dim))
                self.modulations.append(nn.Parameter(torch.ones(hidden_dim)))
                self.time_biases.append(TimeAwareBias(hidden_dim, time_encoding_dim))
        
        self.activation = APELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        time_encoding = self.sinusoidal_encoding(time_step)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.num_layers - 1:
                modulation_factor = self.modulations[i] + self.time_biases[i](time_encoding)
                x = x * modulation_factor
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        return x


# ------------------------- Attention Mechanism -------------------------
class Attention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, attention_dim)
        self.key_proj = nn.Linear(input_dim, attention_dim)
        self.value_proj = nn.Linear(input_dim, attention_dim)
        self.out_proj = nn.Linear(attention_dim, input_dim)
        
        # Initialize parameters with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, input_dim] -> [batch, seq_len, attention_dim]
        query = self.query_proj(input)
        key = self.key_proj(input)
        value = self.value_proj(input)

        # Scaled dot-product attention
        attention_weights = F.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5),
            dim=-1
        )
        
        # Apply attention and project back
        context_vector = torch.matmul(attention_weights, value)
        return self.out_proj(context_vector)


# ------------------------- SAC Actor Network -------------------------
class MetaSACActor(nn.Module):
    """Meta Soft Actor-Critic actor network with attention."""
    def __init__(self, config: MetaSACConfig):
        super().__init__()
        self.attention = Attention(config.state_dim, config.attention_dim)
        self.mlp = AdaptiveModulationMLP(
            config.state_dim, 
            config.hidden_dim,
            2 * config.action_dim,
            config.num_mlp_layers,
            config.dropout_rate,
            config.time_encoding_dim
        )
        self.action_dim = config.action_dim

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.attention(x)
        x = x.squeeze(1)
        x = self.mlp(x, time_step)
        
        mu, log_sigma = x[:, :self.action_dim], x[:, self.action_dim:]
        
        # Add checks for NaN values
        if torch.isnan(mu).any() or torch.isnan(log_sigma).any():
            logger.warning("NaN detected in actor output (mu or log_sigma)")
            mu = torch.nan_to_num(mu)
            log_sigma = torch.nan_to_num(log_sigma)
            
        return torch.tanh(mu), log_sigma


# ------------------------- SAC Critic Network -------------------------
class MetaSACCritic(nn.Module):
    """Meta Soft Actor-Critic critic network with attention."""
    def __init__(self, config: MetaSACConfig):
        super().__init__()
        combined_dim = config.state_dim + config.action_dim
        self.attention = Attention(combined_dim, config.attention_dim)
        self.mlp = AdaptiveModulationMLP(
            combined_dim,
            config.hidden_dim,
            1,
            config.num_mlp_layers,
            config.dropout_rate,
            config.time_encoding_dim
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.attention(x)
        x = x.squeeze(1)
        return self.mlp(x, time_step)


# ------------------------- Meta Controller -------------------------
class MetaController(nn.Module):
    """Meta-controller for adaptive temperature parameter."""
    def __init__(self, config: MetaSACConfig):
        super().__init__()
        self.mlp = ModernMLP(
            config.meta_input_dim + 2, # Added 2 for mean and variance
            config.hidden_dim,
            1,
            config.num_mlp_layers,
            config.dropout_rate
        )

    def forward(self, x: torch.Tensor, reward_stats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, reward_stats], dim=-1)
        return self.mlp(x)


# ------------------------- Replay Buffer -------------------------
class ReplayBuffer:
    """Experience replay buffer with prioritized sampling."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.epsilon = 1e-6  # Small constant to prevent zero priorities

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max(self.priorities.max(), self.epsilon)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in buffer")
            
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities + self.epsilon

    def __len__(self) -> int:
        return len(self.buffer)


# ------------------------- Meta-SAC Agent -------------------------
class MetaSACAgent(nn.Module):
    """Meta Soft Actor-Critic agent."""
    def __init__(self, config: MetaSACConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Initialize networks
        self.actor = MetaSACActor(config).to(device)
        self.critic1 = MetaSACCritic(config).to(device)
        self.critic2 = MetaSACCritic(config).to(device)
        self.critic_target1 = MetaSACCritic(config).to(device)
        self.critic_target2 = MetaSACCritic(config).to(device)
        
        # Copy weights
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        
        # Initialize meta-controller
        self.meta_controller = MetaController(config).to(device)
        
        # Setup optimizers with gradient clipping
        self.setup_optimizers()
        
        # Initialize alpha parameter
        self.alpha = nn.Parameter(torch.tensor(config.alpha, device=device))
        self.target_entropy = -torch.prod(torch.tensor(config.action_dim).float()).to(device)
        
        # Setup tensorboard writer
        self.writer = SummaryWriter()
        self.train_steps = 0
        self.reward_history = []

    def setup_optimizers(self) -> None:
        """Initialize optimizers with learning rate schedulers."""
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.lr)
        self.meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=self.config.meta_lr)
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='min', factor=0.5, patience=5
        )
        self.critic1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic1_optimizer, mode='min', factor=0.5, patience=5
        )
        self.critic2_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic2_optimizer, mode='min', factor=0.5, patience=5
        )

    def select_action(self, state: np.ndarray, time_step: int, eval: bool = False) -> np.ndarray:
        """
        Select an action given the current state and time encoding.
        
        Args:
            state: Current state observation
            time_step: Current time step 
            eval: If True, use deterministic action selection
            
        Returns:
            Selected action
        """
        # Input validation
        if state.shape[-1] != self.config.state_dim:
            raise ValueError(f"Expected state dimension {self.config.state_dim}, got {state.shape[-1]}")
           
        # Handle NaN inputs
        if np.isnan(state).any():
            logger.warning("NaN detected in state input, replacing with zeros")
            state = np.nan_to_num(state)
           
        self.actor.eval() if eval else self.actor.train()
        
        with torch.no_grad():
            # Convert to tensors and move to device
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            time_tensor = torch.tensor(time_step).unsqueeze(0).to(self.device)

            mu, log_sigma = self.actor(state_tensor, time_tensor)
            
            if eval:
                action = mu
            else:
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                action = torch.tanh(dist.rsample())

            # Ensure actions are within bounds
            action = torch.clamp(action, -1.0, 1.0)
            return action.cpu().numpy()[0]

    def compute_q_targets(self, rewards: torch.Tensor, next_states: torch.Tensor, 
                            time_steps: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value targets for critic update.
        
        Args:
            rewards: Batch of rewards
            next_states: Batch of next states
            time_steps: Batch of time steps
            dones: Batch of done flags
            
        Returns:
            Q-value targets
        """
        with torch.no_grad():
            next_actions = []
            next_log_probs = []
            
            # Process each state individually to avoid memory issues
            for ns, ts in zip(next_states, time_steps):
                action = self.select_action(ns.cpu().numpy(), ts.item(), eval=False)
                next_actions.append(action)
                
                # Compute log prob for the selected action
                mu, log_sigma = self.actor(ns.unsqueeze(0), ts.unsqueeze(0))
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                z = dist.rsample()
                action_tensor = torch.tanh(z)
                log_prob = dist.log_prob(z) - torch.log(1 - action_tensor.pow(2) + self.config.epsilon)
                next_log_probs.append(log_prob.sum(-1))

            next_actions = torch.FloatTensor(np.stack(next_actions)).to(self.device)
            next_log_probs = torch.stack(next_log_probs).unsqueeze(-1).to(self.device) # added unsqueeze to make the shape [batch_size,1]

            # Compute target Q-values
            q_target1 = self.critic_target1(next_states, next_actions, time_steps)
            q_target2 = self.critic_target2(next_states, next_actions, time_steps)
            q_target = torch.min(q_target1, q_target2)
            
            # Final target computation
            return rewards + (1.0 - dones) * self.config.gamma * (q_target - self.alpha * next_log_probs)

    def update_critics(self, states: torch.Tensor, actions: torch.Tensor, 
                      time_steps: torch.Tensor, q_targets: torch.Tensor) -> Tuple[float, float]:
        """
        Update critic networks.
        
        Returns:
            Tuple of critic losses
        """
        # Compute current Q-values
        q_value1 = self.critic1(states, actions, time_steps)
        q_value2 = self.critic2(states, actions, time_steps)

        # Compute critic losses
        critic1_loss = F.mse_loss(q_value1, q_targets)
        critic2_loss = F.mse_loss(q_value2, q_targets)

        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.max_grad_norm)
        self.critic1_optimizer.step()

        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.max_grad_norm)
        self.critic2_optimizer.step()

        return critic1_loss.item(), critic2_loss.item()

    def update_actor(self, states: torch.Tensor, time_steps: torch.Tensor) -> float:
        """
        Update actor network.
        
        Returns:
            Actor loss value
        """
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
        """
        Update meta-controller network.
        
        Returns:
            Meta-controller loss value
        """
        # compute reward statistics
        mean = torch.mean(rewards, dim=0, keepdim=True)
        variance = torch.var(rewards, dim=0, keepdim=True)
        reward_stats = torch.cat([mean, variance], dim=-1).to(self.device)
        
        # Repeat the reward_stats along the batch size dimension to match meta_input
        batch_size = meta_input.size(0)
        reward_stats = reward_stats.repeat(batch_size, 1)
        
        meta_output = self.meta_controller(meta_input, reward_stats)
        alpha_target_loss = -(meta_output - torch.log(self.alpha)) * (log_probs + self.target_entropy).detach().mean()

        self.meta_optimizer.zero_grad()
        alpha_target_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_controller.parameters(), self.config.max_grad_norm)
        self.meta_optimizer.step()

        return alpha_target_loss.item()

    def soft_update(self, target_network: nn.Module, source_network: nn.Module) -> None:
        """Soft update of target network parameters."""
        try:
            for target_param, param in zip(target_network.parameters(), source_network.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
                )
        except Exception as e:
            logger.error(f"Failed to update target network: {str(e)}")
            raise

    def update_params(self, replay_buffer: ReplayBuffer, meta_input: np.ndarray, 
                     time_memory: List[int]) -> Dict[str, float]:
        """
        Update all network parameters.
        
        Returns:
            Dictionary containing all loss values
        """
        if len(replay_buffer) < self.config.batch_size:
            return {}

        # Sample from replay buffer
        try:
            batch, indices, weights = replay_buffer.sample(self.config.batch_size)
        except ValueError as e:
            logger.error(f"Failed to sample from replay buffer: {str(e)}")
            return {}

        # Prepare batch data
        states = torch.FloatTensor(np.stack([item[0] for item in batch])).to(self.device)
        actions = torch.FloatTensor(np.stack([item[1] for item in batch])).to(self.device)
        rewards = torch.FloatTensor(np.stack([item[2] for item in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack([item[3] for item in batch])).to(self.device)
        dones = torch.FloatTensor(np.stack([item[4] for item in batch])).unsqueeze(1).to(self.device)
        meta_input = torch.FloatTensor(meta_input).unsqueeze(0).to(self.device)
        time_steps = torch.tensor(time_memory).to(self.device)
        
        # Compute targets for critics
        q_targets = self.compute_q_targets(rewards, next_states, time_steps, dones)

        # Update critics
        critic1_loss, critic2_loss = self.update_critics(states, actions, time_steps, q_targets)
        
        # Update actor
        actor_loss = self.update_actor(states, time_steps)

        # Update meta-controller and alpha
        mu, log_sigma = self.actor(states, time_steps)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        actions = torch.tanh(z)
        log_probs = dist.log_prob(z) - torch.log(1 - actions.pow(2) + self.config.epsilon)
        log_probs = log_probs.sum(-1, keepdim=True)
        meta_loss = self.update_meta_controller(meta_input, log_probs, rewards)  # Send the actual rewards

        # Update target networks
        self.soft_update(self.critic_target1, self.critic1)
        self.soft_update(self.critic_target2, self.critic2)

        # Update learning rates
        self.actor_scheduler.step(actor_loss)
        self.critic1_scheduler.step(critic1_loss)
        self.critic2_scheduler.step(critic2_loss)
            
        # Log to tensorboard
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
        """Save model state."""
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
        """Load model state."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.critic_target1.load_state_dict(checkpoint['critic_target1_state_dict'])
            self.critic_target2.load_state_dict(checkpoint['critic_target2_state_dict'])
            self.meta_controller.load_state_dict(checkpoint['meta_controller_state_dict'])
            self.alpha.data.copy_(torch.tensor(checkpoint['alpha'], device=self.device))
            self.train_steps = checkpoint['train_steps']
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
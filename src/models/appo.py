import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from .nlnn import NLNN
from .reward import CompositeReward, RewardConfig
import threading
from queue import Queue
import time

@dataclass
class APPOConfig:
    """Configuration for APPO algorithm"""
    # Network architecture
    input_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 4
    
    # APPO parameters
    num_workers: int = 12  # Optimized for Ryzen 9 7900X
    batch_size: int = 512  # Optimized for RTX 3070
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    
    # Optimization
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    mini_batch_size: int = 64
    
    # Experience replay
    buffer_size: int = 10000
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    
    # Auxiliary tasks
    use_auxiliary: bool = True
    aux_weight: float = 0.1

class AsyncPPOAgent(nn.Module):
    """Asynchronous Proximal Policy Optimization with hardware optimizations"""
    
    def __init__(self, config: APPOConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.backbone = NLNN(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers
        ).to(self.device)
        
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 3)  # Buy, Sell, Hold
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        ).to(self.device)
        
        if config.use_auxiliary:
            self.auxiliary_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.input_dim)
            ).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.actor.parameters()),
            lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.critic.parameters()),
            lr=config.critic_lr
        )
        
        # Initialize mixed precision training
        self.scaler = GradScaler()
        
        # Initialize experience replay
        self.replay_buffer = PrioritizedReplayBuffer(
            config.buffer_size,
            config.priority_alpha,
            config.priority_beta
        )
        
        # Initialize worker threads
        self.workers = [
            PPOWorker(i, self, Queue())
            for i in range(config.num_workers)
        ]
        
        # Compile model for faster execution
        self.trace_model()
    
    def trace_model(self):
        """JIT compile the model for faster execution"""
        self.backbone = torch.compile(self.backbone)
        self.actor = torch.compile(self.actor)
        self.critic = torch.compile(self.critic)
        if self.config.use_auxiliary:
            self.auxiliary_head = torch.compile(self.auxiliary_head)
    
    @torch.compile
    def forward(self, state: torch.Tensor,
                hidden_states: Optional[List[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass with mixed precision"""
        with autocast():
            # Get features from backbone
            features, new_hidden = self.backbone(state, hidden_states)
            
            # Get action logits and value estimate
            action_logits = self.actor(features)
            value = self.critic(features)
            
            # Get auxiliary prediction if enabled
            if self.config.use_auxiliary:
                aux_pred = self.auxiliary_head(features)
            else:
                aux_pred = None
            
            return action_logits, value, new_hidden, aux_pred
    
    def select_action(self, state: torch.Tensor,
                     hidden_states: List[torch.Tensor],
                     deterministic: bool = False) -> Tuple[torch.Tensor, float, List[torch.Tensor]]:
        """Select action using the current policy"""
        with torch.no_grad():
            action_logits, value, new_hidden, _ = self(state, hidden_states)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch

    def update_minibatch(self, states: torch.Tensor,
                        actions: torch.Tensor,
                        returns: torch.Tensor,
                        advantages: torch.Tensor):
        """Update policy and value function on a single minibatch"""
        with autocast(enabled=self.config.use_amp):
            # Get current policy and value predictions
            new_log_probs, new_values, entropy = self.evaluate_actions(states, actions)
            
            # Calculate policy loss with clipping
            ratio = torch.exp(new_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss with clipping
            value_pred_clipped = values + torch.clamp(
                new_values - values,
                -self.config.clip_ratio,
                self.config.clip_ratio
            )
            value_losses = (new_values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
            
            # Calculate auxiliary loss if enabled
            if self.config.use_auxiliary:
                aux_pred = self.auxiliary_head(features)
                aux_loss = F.mse_loss(aux_pred, next_states)
            else:
                aux_loss = 0
            
            # Combine losses
            total_loss = (
                policy_loss
                - self.config.entropy_coef * entropy
                + self.config.value_loss_coef * value_loss
                + self.config.aux_weight * aux_loss
            )
            
            # Optimize using mixed precision
            if self.config.use_amp:
                self.scaler.scale(total_loss).backward()
                
                # Clip gradients
                if self.config.max_grad_norm > 0:
                    for param_group in [
                        list(self.backbone.parameters()),
                        list(self.actor.parameters()),
                        list(self.critic.parameters())
                    ]:
                        self.scaler.unscale_(self.actor_optimizer)
                        torch.nn.utils.clip_grad_norm_(param_group, self.config.max_grad_norm)
                
                self.scaler.step(self.actor_optimizer)
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.config.max_grad_norm > 0:
                    for param_group in [
                        list(self.backbone.parameters()),
                        list(self.actor.parameters()),
                        list(self.critic.parameters())
                    ]:
                        torch.nn.utils.clip_grad_norm_(param_group, self.config.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item(),
                'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            }
    
    def train(self, num_steps: int):
        """Main training loop"""
        # Start worker threads
        for worker in self.workers:
            worker.start()
        
        total_steps = 0
        episode_rewards = []
        metrics_history = []
        
        try:
            while total_steps < num_steps:
                # Collect experiences from workers
                experiences = []
                while len(experiences) < self.config.batch_size:
                    if not self.experience_queue.empty():
                        exp = self.experience_queue.get()
                        experiences.append(exp)
                        total_steps += 1
                
                # Update policy
                metrics = self.update(experiences)
                metrics_history.append(metrics)
                
                # Log progress
                if total_steps % 1000 == 0:
                    avg_metrics = {
                        k: np.mean([m[k] for m in metrics_history[-1000:]])
                        for k in metrics_history[0].keys()
                    }
                    logger.info(f"Step {total_steps}/{num_steps} - Metrics: {avg_metrics}")
        
        finally:
            # Clean up workers
            for worker in self.workers:
                worker.join()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'auxiliary_state_dict': self.auxiliary_head.state_dict() if self.config.use_auxiliary else None,
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        if self.config.use_auxiliary and checkpoint['auxiliary_state_dict']:
            self.auxiliary_head.load_state_dict(checkpoint['auxiliary_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

if __name__ == "__main__":
    # Example usage
    config = APPOConfig(
        input_dim=64,
        hidden_dim=256,
        num_layers=4,
        num_workers=12,  # Using all cores of Ryzen 9 7900X
        batch_size=512,  # Optimized for RTX 3070 8GB VRAM
        use_amp=True,    # Enable automatic mixed precision
        tensor_parallel=True  # Enable tensor parallelization
    )
    
    agent = AsyncPPOAgent(config)
    
    # Train the agent
    agent.train(num_steps=1000000)
    
    # Save checkpoint
    agent.save_checkpoint("appo_checkpoint.pt")

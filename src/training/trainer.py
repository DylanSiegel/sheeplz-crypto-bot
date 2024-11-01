import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path
import wandb
from loguru import logger

from src.config.config import TradingConfig
from src.models.nlnn import ActorCritic
from src.data.features import FeatureExtractor
from src.env.bybit_env import BybitFuturesEnv
from src.utils.risk_manager import RiskManager

class Trainer:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.env = BybitFuturesEnv(config)
        self.feature_extractor = FeatureExtractor(config)
        self.risk_manager = RiskManager(config)
        
        # Initialize model and optimizer
        self.model = ActorCritic(
            input_size=config.model.feature_dim,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.model.learning_rate
        )
        
        # Initialize mixed precision training
        self.scaler = GradScaler()
        
        # Initialize experience buffer
        self.buffer_size = config.training.update_interval
        self.clear_buffers()
        
        # Setup wandb logging
        wandb.init(
            project="nlnn-trading",
            config=config.dict(),
            name=f"train_{int(time.time())}"
        )
    
    def clear_buffers(self):
        """Reset experience buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.hidden_states = []
        self.masks = []
    
    @torch.compile
    def compute_gae(self,
                    rewards: torch.Tensor,
                    values: torch.Tensor,
                    masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        running_return = 0
        running_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t < len(rewards) - 1:
                next_value = values[t + 1]
            else:
                next_value = 0
                
            running_return = rewards[t] + self.config.training.gamma * running_return * masks[t]
            running_tderror = (
                rewards[t] + 
                self.config.training.gamma * next_value * masks[t] - 
                values[t]
            )
            running_advantage = (
                running_tderror + 
                self.config.training.gamma * 
                self.config.training.gae_lambda * 
                running_advantage * 
                masks[t]
            )
            
            returns[t] = running_return
            advantages[t] = running_advantage
            
        return returns, advantages
    
    def train_step(self) -> Dict[str, float]:
        """Perform a single training step"""
        # Convert buffers to tensors
        states = torch.cat(self.states)
        actions = torch.tensor(self.actions, device=self.device)
        rewards = torch.tensor(self.rewards, device=self.device)
        old_values = torch.cat(self.values)
        old_log_probs = torch.cat(self.log_probs)
        masks = torch.tensor(self.masks, device=self.device)
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(rewards, old_values, masks)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare batches
        batch_size = len(states) // self.config.training.num_minibatches
        indices = np.arange(len(states))
        
        # Training metrics
        metrics = {
            "value_loss": 0,
            "policy_loss": 0,
            "entropy_loss": 0,
            "total_loss": 0
        }
        
        # Perform multiple epochs of training
        for _ in range(self.config.training.num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forwar
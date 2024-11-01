import torch
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch.nn.functional as F
from torch.cuda.amp import autocast

@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    # PnL Rewards
    pnl_scale: float = 1.0
    realized_pnl_weight: float = 0.7
    unrealized_pnl_weight: float = 0.3
    
    # Risk-Adjusted Rewards
    use_sharpe: bool = True
    sharpe_window: int = 100
    risk_free_rate: float = 0.0
    
    # Trading Behavior Rewards
    timing_scale: float = 0.3
    position_scale: float = 0.2
    spread_scale: float = 0.1
    
    # Penalties
    oversizing_penalty: float = -0.5
    overtrading_penalty: float = -0.3
    drawdown_penalty: float = -0.4
    
    # Exploration Rewards
    curiosity_scale: float = 0.1
    novelty_threshold: float = 0.1

class BaseReward(ABC):
    """Abstract base class for reward functions"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def calculate(self, state: Dict[str, torch.Tensor], 
                 action: torch.Tensor, 
                 next_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

class PnLReward(BaseReward):
    """Profit and Loss based reward"""
    
    @torch.compile
    def calculate(self, state: Dict[str, torch.Tensor],
                 action: torch.Tensor,
                 next_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast():
            realized_pnl = next_state['realized_pnl']
            unrealized_pnl = next_state['unrealized_pnl']
            
            pnl_reward = (
                self.config.realized_pnl_weight * realized_pnl +
                self.config.unrealized_pnl_weight * unrealized_pnl
            ) * self.config.pnl_scale
            
            return pnl_reward

class SharpeReward(BaseReward):
    """Sharpe ratio based reward"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.returns_history = torch.zeros(
            (config.sharpe_window,),
            device=self.device
        )
        self.current_idx = 0
    
    @torch.compile
    def calculate(self, state: Dict[str, torch.Tensor],
                 action: torch.Tensor,
                 next_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast():
            # Update returns history
            current_return = (
                next_state['total_pnl'] - state['total_pnl']
            ) / state['account_balance']
            
            self.returns_history[self.current_idx] = current_return
            self.current_idx = (self.current_idx + 1) % self.config.sharpe_window
            
            # Calculate Sharpe ratio
            if self.current_idx >= self.config.sharpe_window:
                excess_returns = self.returns_history - self.config.risk_free_rate
                sharpe = (
                    torch.mean(excess_returns) /
                    (torch.std(excess_returns) + 1e-7)
                ) * torch.sqrt(torch.tensor(252.0, device=self.device))  # Annualized
                
                return sharpe
            
            return torch.tensor(0.0, device=self.device)

class BehaviorReward(BaseReward):
    """Trading behavior based reward"""
    
    @torch.compile
    def calculate(self, state: Dict[str, torch.Tensor],
                 action: torch.Tensor,
                 next_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast():
            # Timing reward based on price movement direction
            price_movement = next_state['close_price'] - state['close_price']
            timing_reward = torch.sign(price_movement) * torch.sign(action - 1)  # -1 for sell, 0 for hold, 1 for buy
            timing_reward *= self.config.timing_scale
            
            # Position sizing reward
            position_size = torch.abs(next_state['position_size'])
            optimal_size = state['account_balance'] * state['volatility']
            position_reward = -torch.abs(position_size - optimal_size) * self.config.position_scale
            
            # Spread cost penalty
            spread_cost = state['bid_ask_spread'] * torch.abs(action - 1)
            spread_penalty = -spread_cost * self.config.spread_scale
            
            return timing_reward + position_reward + spread_penalty

class RiskPenalty(BaseReward):
    """Risk-based penalties"""
    
    @torch.compile
    def calculate(self, state: Dict[str, torch.Tensor],
                 action: torch.Tensor,
                 next_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast():
            penalties = torch.tensor(0.0, device=self.device)
            
            # Oversizing penalty
            max_position = state['account_balance'] * state['max_position_size']
            if torch.abs(next_state['position_size']) > max_position:
                penalties += self.config.oversizing_penalty
            
            # Overtrading penalty
            if state['trade_count'] > state['optimal_trade_count']:
                penalties += self.config.overtrading_penalty
            
            # Drawdown penalty
            current_drawdown = (state['peak_equity'] - next_state['account_balance']) / state['peak_equity']
            if current_drawdown > state['max_drawdown']:
                penalties += self.config.drawdown_penalty
            
            return penalties

class CuriosityReward(BaseReward):
    """Intrinsic motivation through curiosity"""
    
    def __init__(self, config: RewardConfig, state_dim: int):
        super().__init__(config)
        self.state_history = torch.zeros(
            (1000, state_dim),
            device=self.device
        )
        self.current_idx = 0
    
    @torch.compile
    def calculate(self, state: Dict[str, torch.Tensor],
                 action: torch.Tensor,
                 next_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        with autocast():
            state_tensor = torch.cat([
                torch.tensor(list(state.values()), device=self.device)
            ])
            
            # Update state history
            self.state_history[self.current_idx] = state_tensor
            self.current_idx = (self.current_idx + 1) % 1000
            
            # Calculate novelty as distance to nearest neighbor
            distances = torch.norm(
                self.state_history - state_tensor.unsqueeze(0),
                dim=1
            )
            novelty = torch.min(distances[distances > 0])
            
            # Scale novelty reward
            novelty_reward = torch.where(
                novelty > self.config.novelty_threshold,
                self.config.curiosity_scale,
                torch.tensor(0.0, device=self.device)
            )
            
            return novelty_reward

class CompositeReward:
    """Combines multiple reward components"""
    
    def __init__(self, config: RewardConfig, state_dim: int):
        self.components = {
            'pnl': PnLReward(config),
            'sharpe': SharpeReward(config),
            'behavior': BehaviorReward(config),
            'risk': RiskPenalty(config),
            'curiosity': CuriosityReward(config, state_dim)
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @torch.compile
    def calculate_reward(self,
                        state: Dict[str, torch.Tensor],
                        action: torch.Tensor,
                        next_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate combined reward from all components"""
        with autocast():
            rewards = {}
            
            # Calculate rewards in parallel if on GPU
            if self.device.type == 'cuda':
                streams = [torch.cuda.Stream() for _ in self.components]
                
                for (name, component), stream in zip(self.components.items(), streams):
                    with torch.cuda.stream(stream):
                        rewards[name] = component.calculate(state, action, next_state)
                
                torch.cuda.synchronize()
            else:
                for name, component in self.components.items():
                    rewards[name] = component.calculate(state, action, next_state)
            
            # Combine rewards
            total_reward = sum(rewards.values())
            
            # Clip final reward for stability
            return torch.clamp(total_reward, -10.0, 10.0)
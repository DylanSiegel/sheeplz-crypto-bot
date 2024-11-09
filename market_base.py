from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import numpy as np
import pandas as pd
from enum import Enum

class Action(Enum):
    """Trading action enumeration"""
    HOLD = 0
    BUY = 1
    SELL = 2

@dataclass
class MarketState:
    """Container for market state information with validation"""
    encoded_state: torch.Tensor
    regime_label: int
    current_price: float
    timestamp: pd.Timestamp
    metrics: Dict[str, float]
    
    def __post_init__(self):
        """Validate state attributes after initialization"""
        if not isinstance(self.encoded_state, torch.Tensor):
            raise TypeError("encoded_state must be a torch.Tensor")
        if not isinstance(self.regime_label, (int, np.integer)):
            raise TypeError("regime_label must be an integer")
        if not isinstance(self.current_price, (float, np.floating)):
            raise TypeError("current_price must be a float")
        if not isinstance(self.timestamp, pd.Timestamp):
            raise TypeError("timestamp must be a pd.Timestamp")
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dictionary")
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor format for DRL"""
        return self.encoded_state
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format"""
        return {
            'encoded_state': self.encoded_state.numpy(),
            'regime_label': self.regime_label,
            'current_price': self.current_price,
            'timestamp': self.timestamp,
            'metrics': self.metrics
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> 'MarketState':
        """Create MarketState from dictionary"""
        return cls(
            encoded_state=torch.tensor(state_dict['encoded_state']),
            regime_label=state_dict['regime_label'],
            current_price=state_dict['current_price'],
            timestamp=pd.Timestamp(state_dict['timestamp']),
            metrics=state_dict['metrics']
        )

class RewardCalculator:
    """Calculate DRL rewards based on market state and actions with regime-aware adjustments"""
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        holding_cost: float = 0.0001,
        volatility_penalty: float = 0.1,
        regime_multipliers: Optional[Dict[int, float]] = None
    ):
        """
        Initialize reward calculator with costs and penalties
        
        Args:
            transaction_cost: Cost per transaction as percentage
            holding_cost: Cost of holding position per step
            volatility_penalty: Penalty factor for volatility
            regime_multipliers: Dict mapping regime IDs to reward multipliers
        """
        self._validate_costs(transaction_cost, holding_cost, volatility_penalty)
        self.transaction_cost = transaction_cost
        self.holding_cost = holding_cost
        self.volatility_penalty = volatility_penalty
        
        # Set default regime multipliers if none provided
        self.regime_multipliers = regime_multipliers or {
            0: 0.8,  # High volatility - reduce reward
            1: 1.0,  # Normal trading
            2: 1.2,  # Trending market - increase reward
        }
        
        # Initialize metrics tracking
        self.reset_metrics()
    
    def _validate_costs(
        self,
        transaction_cost: float,
        holding_cost: float,
        volatility_penalty: float
    ) -> None:
        """Validate cost parameters"""
        if not 0 <= transaction_cost <= 0.1:
            raise ValueError("transaction_cost must be between 0 and 0.1")
        if not 0 <= holding_cost <= 0.01:
            raise ValueError("holding_cost must be between 0 and 0.01")
        if not 0 <= volatility_penalty <= 1.0:
            raise ValueError("volatility_penalty must be between 0 and 1.0")
    
    def reset_metrics(self) -> None:
        """Reset accumulated metrics"""
        self.total_rewards = 0.0
        self.total_costs = 0.0
        self.total_penalties = 0.0
        self.rewards_by_regime = {}
        self.action_counts = {action: 0 for action in Action}
    
    def calculate_reward(
        self,
        current_state: MarketState,
        next_state: MarketState,
        action: Union[Action, int],
        position_size: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward for a single state transition
        
        Args:
            current_state: Current market state
            next_state: Next market state
            action: Trading action (HOLD=0, BUY=1, SELL=2)
            position_size: Size of position as percentage of portfolio
            
        Returns:
            reward: Calculated reward
            metrics: Dictionary of reward components
        """
        # Validate inputs
        if isinstance(action, int):
            action = Action(action)
        if not isinstance(action, Action):
            raise ValueError(f"Invalid action: {action}")
        if not 0 <= position_size <= 1:
            raise ValueError("position_size must be between 0 and 1")
        
        # Calculate price change
        price_change = (next_state.current_price - current_state.current_price) / current_state.current_price
        
        # Calculate base reward
        if action == Action.BUY:
            base_reward = price_change * position_size
            costs = self.transaction_cost * position_size
        elif action == Action.SELL:
            base_reward = -price_change * position_size
            costs = self.transaction_cost * position_size
        else:  # HOLD
            base_reward = 0
            costs = self.holding_cost * position_size
        
        # Apply volatility penalty
        volatility = current_state.metrics.get('volatility', 0)
        vol_penalty = -self.volatility_penalty * volatility * position_size
        
        # Get regime multiplier
        regime_multiplier = self.regime_multipliers.get(
            current_state.regime_label,
            1.0
        )
        
        # Calculate final reward
        reward = (base_reward - costs + vol_penalty) * regime_multiplier
        
        # Update metrics
        self.total_rewards += reward
        self.total_costs += costs
        self.total_penalties += abs(vol_penalty)
        self.action_counts[action] += 1
        
        if current_state.regime_label not in self.rewards_by_regime:
            self.rewards_by_regime[current_state.regime_label] = []
        self.rewards_by_regime[current_state.regime_label].append(reward)
        
        metrics = {
            'base_reward': base_reward,
            'costs': costs,
            'vol_penalty': vol_penalty,
            'regime_multiplier': regime_multiplier,
            'total_reward': reward,
            'price_change': price_change,
            'volatility': volatility
        }
        
        return reward, metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of accumulated metrics"""
        summary = {
            'total_rewards': self.total_rewards,
            'total_costs': self.total_costs,
            'total_penalties': self.total_penalties,
            'action_counts': {action.name: count for action, count in self.action_counts.items()},
            'rewards_by_regime': {
                regime: {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'count': len(rewards)
                }
                for regime, rewards in self.rewards_by_regime.items()
            }
        }
        return summary

class DataValidationMixin:
    """Mixin class for data validation methods"""
    
    @staticmethod
    def validate_market_data(
        data: np.ndarray,
        min_sequence_length: int,
        expected_features: int
    ) -> None:
        """Validate market data array"""
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array")
        
        if data.ndim != 3:
            raise ValueError(f"data must be 3D, got shape {data.shape}")
            
        if data.shape[1] < min_sequence_length:
            raise ValueError(
                f"Sequence length must be at least {min_sequence_length}, "
                f"got {data.shape[1]}"
            )
            
        if data.shape[2] < expected_features:
            raise ValueError(
                f"Expected at least {expected_features} features, "
                f"got {data.shape[2]}"
            )
    
    @staticmethod
    def validate_timestamps(
        timestamps: pd.DatetimeIndex,
        n_samples: int
    ) -> None:
        """Validate timestamp index"""
        if not isinstance(timestamps, pd.DatetimeIndex):
            raise TypeError("timestamps must be a pandas DatetimeIndex")
            
        if len(timestamps) != n_samples:
            raise ValueError(
                f"Number of timestamps ({len(timestamps)}) must match "
                f"number of samples ({n_samples})"
            )
        
        if timestamps.freq is None:
            if pd.infer_freq(timestamps) is None:
                raise ValueError(
                    "timestamps must have a frequency or regular interval"
                )
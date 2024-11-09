import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import logging
from torch.cuda import amp
from market_base import Action, MarketState
from optimized_market import OptimizedRewardCalculator

logger = logging.getLogger(__name__)

@dataclass
class TradingEnvConfig:
    """Configuration for trading environment"""
    initial_balance: float = 10000.0
    max_position_size: float = 1.0
    transaction_cost: float = 0.001
    history_length: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp: bool = True  # Use automatic mixed precision
    batch_size: int = 512  # Optimized for 8GB VRAM

class TradingEnvironment:
    """GPU-accelerated trading environment with batched operations"""
    
    def __init__(
        self,
        market_states: List[MarketState],
        config: Optional[TradingEnvConfig] = None
    ):
        self.config = config or TradingEnvConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize reward calculator
        self.reward_calc = OptimizedRewardCalculator(
            transaction_cost=self.config.transaction_cost,
            device=self.config.device
        )
        
        # Convert market states to tensors and move to GPU
        self.states = self._prepare_market_states(market_states)
        self.current_step = 0
        self.history = deque(maxlen=self.config.history_length)
        
        # Initialize portfolio state
        self._initialize_portfolio()
        
        # Enable automatic mixed precision if configured
        self.scaler = amp.GradScaler() if self.config.use_amp else None
        
        logger.info(f"Trading environment initialized with {len(market_states)} states")
    
    def _prepare_market_states(self, market_states: List[MarketState]) -> Dict[str, torch.Tensor]:
        """Prepare market states for GPU processing"""
        encoded_states = []
        prices = []
        regime_labels = []
        
        # Process in batches to manage memory
        for i in range(0, len(market_states), self.config.batch_size):
            batch = market_states[i:i + self.config.batch_size]
            encoded_states.extend([state.encoded_state for state in batch])
            prices.extend([state.current_price for state in batch])
            regime_labels.extend([state.regime_label for state in batch])
        
        return {
            'encoded_states': torch.stack(encoded_states).to(self.device),
            'prices': torch.tensor(prices, dtype=torch.float32).to(self.device),
            'regime_labels': torch.tensor(regime_labels, dtype=torch.long).to(self.device)
        }
    
    def _initialize_portfolio(self):
        """Initialize portfolio state tensors on GPU"""
        self.balance = torch.tensor(self.config.initial_balance, 
                                  device=self.device, 
                                  dtype=torch.float32)
        self.position_size = torch.tensor(0.0, 
                                        device=self.device, 
                                        dtype=torch.float32)
        self.position_value = torch.tensor(0.0, 
                                         device=self.device, 
                                         dtype=torch.float32)
    
    @torch.no_grad()
    def step(self, action: int) -> Tuple[MarketState, float, bool, Dict]:
        """Execute one environment step with GPU acceleration"""
        if self.current_step >= len(self.states['encoded_states']) - 1:
            return self._get_current_state(), 0.0, True, {}
        
        current_state = self._get_current_state()
        next_state = self._get_next_state()
        
        # Use automatic mixed precision for calculations if configured
        with amp.autocast() if self.config.use_amp else nullcontext():
            # Calculate reward and execute trade
            reward, metrics = self.reward_calc.calculate_reward(
                current_state,
                next_state,
                action,
                float(self.position_size)
            )
            
            # Update portfolio based on action
            self._execute_trade(action, current_state, next_state)
        
        # Update state
        self.current_step += 1
        
        # Record state in history
        self.history.append({
            'step': self.current_step,
            'action': action,
            'reward': reward,
            'balance': float(self.balance),
            'position_size': float(self.position_size),
            'price': float(self.states['prices'][self.current_step])
        })
        
        done = self.current_step >= len(self.states['encoded_states']) - 1
        info = {
            'metrics': metrics,
            'portfolio_value': float(self.balance + self.position_value)
        }
        
        return next_state, reward, done, info
    
    @torch.no_grad()
    def _execute_trade(
        self,
        action: int,
        current_state: MarketState,
        next_state: MarketState
    ) -> None:
        """Execute trading action with GPU-accelerated calculations"""
        current_price = self.states['prices'][self.current_step]
        
        if action == Action.BUY.value and self.position_size < self.config.max_position_size:
            # Calculate maximum purchasable amount
            max_purchase = (self.balance * (1 - self.config.transaction_cost)) / current_price
            purchase_size = min(
                self.config.max_position_size - self.position_size,
                max_purchase
            )
            
            if purchase_size > 0:
                cost = purchase_size * current_price * (1 + self.config.transaction_cost)
                self.balance -= cost
                self.position_size += purchase_size
                
        elif action == Action.SELL.value and self.position_size > 0:
            # Calculate sale proceeds
            sale_proceeds = (self.position_size * current_price * 
                           (1 - self.config.transaction_cost))
            self.balance += sale_proceeds
            self.position_size = torch.tensor(0.0, device=self.device)
        
        # Update position value
        self.position_value = self.position_size * self.states['prices'][self.current_step]
    
    def reset(self) -> MarketState:
        """Reset the environment to initial state"""
        self.current_step = 0
        self._initialize_portfolio()
        self.history.clear()
        return self._get_current_state()
    
    def _get_current_state(self) -> MarketState:
        """Get current market state"""
        return MarketState(
            encoded_state=self.states['encoded_states'][self.current_step],
            regime_label=int(self.states['regime_labels'][self.current_step]),
            current_price=float(self.states['prices'][self.current_step]),
            timestamp=None,  # Timestamp not needed for training
            metrics=self._get_state_metrics()
        )
    
    def _get_next_state(self) -> MarketState:
        """Get next market state"""
        next_step = self.current_step + 1
        return MarketState(
            encoded_state=self.states['encoded_states'][next_step],
            regime_label=int(self.states['regime_labels'][next_step]),
            current_price=float(self.states['prices'][next_step]),
            timestamp=None,
            metrics=self._get_state_metrics()
        )
    
    def _get_state_metrics(self) -> Dict[str, float]:
        """Calculate current state metrics"""
        return {
            'portfolio_value': float(self.balance + self.position_value),
            'position_size': float(self.position_size),
            'balance': float(self.balance)
        }

class nullcontext:
    """Context manager that does nothing"""
    def __enter__(self): return None
    def __exit__(self, *excinfo): pass
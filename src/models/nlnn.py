import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
import logging
import time
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class TradingConfig:
    """Configuration for trading parameters and risk management"""
    max_position_size: float = 0.1  # Maximum position size as fraction of capital
    stop_loss_pct: float = 0.02     # Stop loss percentage
    take_profit_pct: float = 0.04   # Take profit percentage
    max_leverage: float = 5.0       # Maximum allowed leverage
    min_trade_interval: int = 5     # Minimum intervals between trades
    rolling_window_size: int = 100  # Size of rolling window for feature calculation

class FeatureExtractor:
    """Extracts and normalizes trading features"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.price_history = deque(maxlen=config.rolling_window_size)
        self.volume_history = deque(maxlen=config.rolling_window_size)
        
    def calculate_features(self, market_data: Dict) -> np.ndarray:
        """Calculate normalized feature vector from market data"""
        # Add new data points
        self.price_history.append(market_data['close_price'])
        self.volume_history.append(market_data['volume'])
        
        if len(self.price_history) < 2:
            return np.zeros(10)  # Return zero features if insufficient history
            
        # Calculate technical features
        returns = np.diff(self.price_history) / np.array(list(self.price_history)[:-1])
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
        rsi = self._calculate_rsi(list(self.price_history))
        
        # Normalize and combine features
        features = np.array([
            self._normalize(market_data['close_price'], self.price_history),
            self._normalize(market_data['volume'], self.volume_history),
            volatility,
            rsi / 100.0,  # RSI is already 0-100
            market_data['bid_ask_spread'] / market_data['close_price'],
            market_data['funding_rate'],
            market_data['open_interest'] / np.mean(self.volume_history),
            market_data['leverage_ratio'] / self.config.max_leverage,
            market_data['market_depth_ratio'],
            market_data['taker_buy_ratio']
        ])
        
        return np.clip(features, -1, 1)  # Ensure all features are in [-1, 1]
    
    def _normalize(self, value: float, history: deque) -> float:
        """Min-max normalization using recent history"""
        if len(history) < 2:
            return 0
        min_val = min(history)
        max_val = max(history)
        if min_val == max_val:
            return 0
        return (value - min_val) / (max_val - min_val) * 2 - 1
    
    def _calculate_rsi(self, prices: List[float], periods: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < periods + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.clip(deltas, 0, None)
        losses = -np.clip(deltas, None, 0)
        
        avg_gain = np.mean(gains[-periods:])
        avg_loss = np.mean(losses[-periods:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class RiskManager:
    """Handles position sizing and risk management"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.current_position = 0
        self.last_trade_time = 0
        
    def validate_action(self, action: int, current_price: float, 
                       account_balance: float) -> Tuple[bool, str]:
        """Validate if an action can be executed based on risk parameters"""
        current_time = time.time()
        
        # Check trading frequency
        if current_time - self.last_trade_time < self.config.min_trade_interval:
            return False, "Trading too frequently"
            
        # Check position sizes
        new_position = self._calculate_position_size(action, current_price, account_balance)
        if abs(new_position) > self.config.max_position_size * account_balance:
            return False, "Position size exceeds maximum"
            
        return True, ""
        
    def _calculate_position_size(self, action: int, current_price: float,
                               account_balance: float) -> float:
        """Calculate appropriate position size based on risk parameters"""
        base_size = account_balance * self.config.max_position_size
        
        # Scale position size based on market volatility (simplified)
        # In practice, you'd want more sophisticated volatility adjustment
        volatility_scalar = 0.5  # Could be dynamically calculated
        position_size = base_size * volatility_scalar
        
        # Adjust for action type (0: Buy, 1: Sell, 2: Hold)
        if action == 2:  # Hold
            return self.current_position
        elif action == 1:  # Sell
            return -position_size
        else:  # Buy
            return position_size

class N_LNN(nn.Module):
    """Enhanced n-LNN implementation with additional features"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input transformation
        self.input_transform = nn.Linear(input_size, hidden_size)
        
        # Learnable parameters
        self.W_i = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, hidden_size)) 
            for _ in range(num_layers)
        ])
        self.W_r = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, hidden_size))
            for _ in range(num_layers)
        ])
        
        # Eigen learning rates and scaling factors
        self.lambda_i = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_size))
            for _ in range(num_layers)
        ])
        self.lambda_r = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_size))
            for _ in range(num_layers)
        ])
        
        self.scaling_i = nn.Parameter(torch.ones(1))
        self.scaling_r = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def normalize(self, v: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Hypersphere normalization with numerical stability"""
        norm = torch.norm(v, dim=-1, keepdim=True)
        return v / (norm + epsilon)
    
    def slerp(self, h_t: torch.Tensor, h_new: torch.Tensor,
              alpha: float = 0.5) -> torch.Tensor:
        """Spherical linear interpolation with safety checks"""
        # Ensure inputs are normalized
        h_t = self.normalize(h_t)
        h_new = self.normalize(h_new)
        
        # Calculate cos_theta with numerical stability
        cos_theta = torch.sum(h_t * h_new, dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)
        
        # Handle edge cases
        sin_theta = torch.sin(theta)
        mask = sin_theta.abs() > 1e-7
        
        result = torch.where(
            mask,
            ((torch.sin((1 - alpha) * theta) / sin_theta) * h_t) + 
            ((torch.sin(alpha * theta) / sin_theta) * h_new),
            h_t
        )
        
        return self.normalize(result)
    
    def forward(self, inputs: torch.Tensor, 
                hidden_states: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with multi-layer support and residual connections"""
        # Input projection
        x = self.input_transform(inputs)
        x = self.layer_norm(x)
        
        new_hidden_states = []
        for layer in range(self.num_layers):
            h_t = hidden_states[layer]
            
            # Apply transformations
            i_transform = torch.matmul(x, self.W_i[layer]) * self.lambda_i[layer]
            r_transform = torch.matmul(h_t, self.W_r[layer]) * self.lambda_r[layer]
            
            # Combine transformations
            h_new = self.normalize(
                self.scaling_i * i_transform + 
                self.scaling_r * r_transform
            )
            
            # Apply SLERP
            h_new = self.slerp(h_t, h_new)
            
            # Residual connection if dimensions match
            if x.shape == h_new.shape:
                h_new = h_new + x
                
            h_new = self.layer_norm(h_new)
            h_new = self.dropout(h_new)
            
            new_hidden_states.append(h_new)
            x = h_new  # Use as input for next layer
            
        return x, new_hidden_states

class Agent(nn.Module):
    """Trading agent with actor-critic architecture"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.n_lnn = N_LNN(input_size, hidden_size, num_layers)
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),  # 3 actions: Buy, Sell, Hold
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=np.sqrt(2))
            module.bias.data.zero_()
            
    def forward(self, state: torch.Tensor, hidden_states: List[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning action probabilities and value estimate"""
        features, new_hidden = self.n_lnn(state, hidden_states)
        
        # Get action probabilities and value estimate
        action_logits = self.actor(features)
        value = self.critic(features)
        
        # Apply action masking if needed (e.g., prevent buying when already long)
        # action_mask = self._get_action_mask(state)
        # action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        
        return F.softmax(action_logits, dim=-1), value, new_hidden
    
    def act(self, state: torch.Tensor, hidden_states: List[torch.Tensor]
            ) -> Tuple[int, float, List[torch.Tensor]]:
        """Select action based on current policy"""
        with torch.no_grad():
            action_probs, value, new_hidden = self(state, hidden_states)
            action = torch.multinomial(action_probs, 1).item()
            
        return action, value.item(), new_hidden

# Example usage:
if __name__ == "__main__":
    # Configuration
    config = TradingConfig()
    feature_extractor = FeatureExtractor(config)
    risk_manager = RiskManager(config)
    
    # Initialize agent
    input_size = 10  # Number of features
    hidden_size = 64
    num_layers = 2
    agent = Agent(input_size, hidden_size, num_layers)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    # Initialize hidden states
    hidden_states = [
        torch.zeros(1, hidden_size) for _ in range(num_layers)
    ]
    
    # Example market data
    market_data = {
        'close_price': 50000.0,
        'volume': 100.0,
        'bid_ask_spread': 1.0,
        'funding_rate': 0.0001,
        'open_interest': 1000.0,
        'leverage_ratio': 2.0,
        'market_depth_ratio': 0.5,
        'taker_buy_ratio': 0.6
    }
    
    # Extract features
    features = feature_extractor.calculate_features(market_data)
    state = torch.FloatTensor(features).unsqueeze(0)
    
    # Get action
    action, value, new_hidden = agent.act(state, hidden_states)
    
    # Validate action
    is_valid, message = risk_manager.validate_action(
        action, market_data['close_price'], account_balance=10000.0
    )
    
    if is_valid:
        logging.info(f"Taking action {action} with value estimate {value}")
    else:
        logging.warning(f"Action rejected: {message}")
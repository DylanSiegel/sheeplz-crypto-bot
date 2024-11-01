import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytest
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class RewardConfig:
    """Configuration for multi-factor reward calculation"""
    # PnL Factors
    pnl_scale: float = 1.0
    realized_weight: float = 0.7
    unrealized_weight: float = 0.3
    
    # Risk Factors
    max_drawdown_penalty: float = -2.0
    overexposure_penalty: float = -1.0
    volatility_scale: float = 0.2
    
    # Trading Factors
    entry_reward: float = 0.1
    exit_reward: float = 0.1
    holding_cost: float = -0.01
    spread_penalty: float = -0.1
    
    # Market Factors
    trend_alignment: float = 0.2
    volume_scale: float = 0.1
    funding_penalty: float = -0.5

class EnhancedReward:
    """Multi-factor reward calculation system"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.price_history = []
        self.position_history = []
        self.pnl_history = []
        self.peak_equity = 0
        
    def calculate_reward(self, 
                        current_state: Dict,
                        previous_state: Dict,
                        action: float,
                        trade_result: Dict) -> Tuple[float, Dict]:
        """Calculate comprehensive reward based on multiple factors"""
        
        # Update histories
        self.price_history.append(current_state['market_data']['last_price'])
        self.position_history.append(current_state['account_data']['position_size'])
        self.pnl_history.append(current_state['account_data']['wallet_balance'])
        
        # Calculate individual reward components
        pnl_reward = self._calculate_pnl_reward(current_state, previous_state)
        risk_reward = self._calculate_risk_reward(current_state)
        trading_reward = self._calculate_trading_reward(current_state, action, trade_result)
        market_reward = self._calculate_market_reward(current_state)
        
        # Combine rewards
        total_reward = (
            pnl_reward +
            risk_reward +
            trading_reward +
            market_reward
        )
        
        # Update peak equity
        self.peak_equity = max(self.peak_equity, current_state['account_data']['wallet_balance'])
        
        # Return reward and components for logging
        components = {
            'pnl_reward': pnl_reward,
            'risk_reward': risk_reward,
            'trading_reward': trading_reward,
            'market_reward': market_reward,
            'total_reward': total_reward
        }
        
        return total_reward, components
    
    def _calculate_pnl_reward(self, current_state: Dict, previous_state: Dict) -> float:
        """Calculate PnL-based reward component"""
        realized_pnl = (
            current_state['account_data']['wallet_balance'] -
            previous_state['account_data']['wallet_balance']
        )
        unrealized_pnl = current_state['account_data']['unrealized_pnl']
        
        return self.config.pnl_scale * (
            self.config.realized_weight * realized_pnl +
            self.config.unrealized_weight * unrealized_pnl
        )
    
    def _calculate_risk_reward(self, current_state: Dict) -> float:
        """Calculate risk-based reward component"""
        # Drawdown penalty
        current_drawdown = (self.peak_equity - current_state['account_data']['wallet_balance']) / self.peak_equity
        drawdown_penalty = self.config.max_drawdown_penalty * current_drawdown
        
        # Position overexposure penalty
        position_size = abs(current_state['account_data']['position_size'])
        max_position = current_state['account_data']['wallet_balance'] * 0.1  # 10% max position
        overexposure_penalty = self.config.overexposure_penalty * max(0, position_size - max_position)
        
        # Volatility adjustment
        if len(self.price_history) > 20:
            returns = np.diff(self.price_history[-20:]) / self.price_history[-21:-1]
            volatility = np.std(returns)
            volatility_adjustment = -self.config.volatility_scale * volatility
        else:
            volatility_adjustment = 0
            
        return drawdown_penalty + overexposure_penalty + volatility_adjustment
    
    def _calculate_trading_reward(self, current_state: Dict, action: float, trade_result: Dict) -> float:
        """Calculate trading behavior-based reward component"""
        # Entry/exit rewards
        position_change = abs(action) > 0.01
        if position_change:
            trading_reward = self.config.entry_reward if abs(action) > 0 else self.config.exit_reward
        else:
            trading_reward = 0
            
        # Holding cost for existing positions
        position_size = abs(current_state['account_data']['position_size'])
        holding_cost = self.config.holding_cost * position_size
        
        # Spread penalty
        spread = (
            current_state['market_data']['best_ask'] -
            current_state['market_data']['best_bid']
        ) / current_state['market_data']['last_price']
        spread_penalty = self.config.spread_penalty * spread * abs(action)
        
        return trading_reward + holding_cost + spread_penalty
    
    def _calculate_market_reward(self, current_state: Dict) -> float:
        """Calculate market condition-based reward component"""
        # Trend alignment reward
        if len(self.price_history) > 20:
            trend = (self.price_history[-1] - self.price_history[-20]) / self.price_history[-20]
            position = current_state['account_data']['position_size']
            trend_reward = self.config.trend_alignment * trend * np.sign(position)
        else:
            trend_reward = 0
            
        # Volume-based reward
        volume_ratio = current_state['market_data']['24h_volume'] / np.mean(self.price_history)
        volume_reward = self.config.volume_scale * np.log1p(volume_ratio)
        
        # Funding rate penalty
        funding_rate = current_state['market_data']['funding_rate']
        position = current_state['account_data']['position_size']
        funding_penalty = self.config.funding_penalty * funding_rate * position
        
        return trend_reward + volume_reward + funding_penalty

class TestBybitEnvironment:
    """Test suite for Bybit trading environment"""
    
    @pytest.fixture
    def env(self):
        """Create test environment"""
        return BybitBTCEnvironment(
            api_key="test_key",
            api_secret="test_secret",
            initial_balance=10000,
            max_position=0.1
        )
    
    @pytest.fixture
    def reward_calculator(self):
        """Create reward calculator"""
        return EnhancedReward(RewardConfig())
    
    def test_environment_initialization(self, env):
        """Test environment initialization"""
        assert env.symbol == "BTCUSDT"
        assert env.initial_balance == 10000
        assert env.current_position == 0
        assert env.entry_price == 0
    
    def test_market_data_fetching(self, env):
        """Test market data fetching"""
        market_data = env._get_market_data()
        required_fields = [
            'last_price', 'mark_price', 'index_price', '24h_volume',
            'funding_rate', 'best_bid', 'best_ask'
        ]
        for field in required_fields:
            assert field in market_data
            assert isinstance(market_data[field], float)
    
    def test_account_data_fetching(self, env):
        """Test account data fetching"""
        account_data = env._get_account_data()
        required_fields = [
            'wallet_balance', 'available_balance', 'position_size',
            'position_value', 'unrealized_pnl'
        ]
        for field in required_fields:
            assert field in account_data
            assert isinstance(account_data[field], float)
    
    def test_trade_execution(self, env):
        """Test trade execution"""
        action = 0.5  # 50% of max position
        trade_result = env._execute_trade(action)
        assert 'orderId' in trade_result
        assert env.current_position > 0
    
    def test_environment_reset(self, env):
        """Test environment reset"""
        observation, info = env.reset()
        assert 'market_data' in observation
        assert 'account_data' in observation
        assert env.current_position == 0
        assert env.account_balance == env.initial_balance
    
    def test_full_episode(self, env, reward_calculator):
        """Test full trading episode"""
        observation, info = env.reset()
        total_rewards = 0
        
        for _ in range(10):  # Run 10 steps
            action = np.random.uniform(-1, 1)  # Random actions
            next_observation, reward, done, truncated, info = env.step(action)
            
            # Calculate enhanced reward
            enhanced_reward, components = reward_calculator.calculate_reward(
                next_observation,
                observation,
                action,
                info['trade_result']
            )
            
            total_rewards += enhanced_reward
            observation = next_observation
            
            if done:
                break
                
        assert isinstance(total_rewards, float)
        env.close()
    
    def test_position_limits(self, env):
        """Test position size limits"""
        # Try to open position larger than max_position
        action = 2.0  # Should be clipped to 1.0
        observation, reward, done, truncated, info = env.step(action)
        assert abs(env.current_position) <= env.max_position * env.account_balance
    
    def test_drawdown_termination(self, env):
        """Test drawdown-based termination"""
        # Simulate large loss
        env.account_balance = env.initial_balance * 0.4  # 60% loss
        observation, reward, done, truncated, info = env.step(0)
        assert done == True

def plot_trading_session(env_history: Dict):
    """Plot trading session results"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot price and positions
    ax1.plot(env_history['prices'], label='Price')
    ax1.set_title('Price and Positions')
    ax1.set_ylabel('Price')
    
    # Add position markers
    for i, pos in enumerate(env_history['positions']):
        if pos > 0:
            ax1.scatter(i, env_history['prices'][i], color='green', marker='^')
        elif pos < 0:
            ax1.scatter(i, env_history['prices'][i], color='red', marker='v')
    
    # Plot PnL
    ax2.plot(env_history['pnl'], label='PnL')
    ax2.set_title('Profit and Loss')
    ax2.set_ylabel('USDT')
    
    # Plot rewards
    ax3.plot(env_history['rewards'], label='Reward')
    ax3.set_title('Rewards')
    ax3.set_ylabel('Reward')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run basic validation
    env = BybitBTCEnvironment(
        api_key="your_testnet_api_key",
        api_secret="your_testnet_api_secret"
    )
    reward_calculator = EnhancedReward(RewardConfig())
    
    # Run test trading session
    observation, info = env.reset()
    history = {
        'prices': [],
        'positions': [],
        'pnl': [],
        'rewards': []
    }
    
    try:
        for _ in range(100):  # Run 100 steps
            # Simple momentum strategy for testing
            returns = np.diff(history['prices'][-20:]) if len(history['prices']) >= 20 else [0]
            action = np.sign(np.mean(returns)) * 0.5  # 50% of max position
            
            # Take action
            next_observation, reward, done, truncated, info = env.step(action)
            
            # Calculate enhanced reward
            enhanced_reward, components = reward_calculator.calculate_reward(
                next_observation,
                observation,
                action,
                info['trade_result']
            )
            
            # Update history
            history['prices'].append(next_observation['market_data'][0])  # Last price
            history['positions'].append(next_observation['account_data'][1])  # Position size
            history['pnl'].append(next_observation['account_data'][0] - env.initial_balance)  # PnL
            history['rewards'].append(enhanced_reward)
            
            observation = next_observation
            
            if done:
                print("Episode finished due to termination condition")
                break
                
    finally:
        env.close()
        
    # Plot results
    plot_trading_session(history)
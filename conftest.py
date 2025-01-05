# File: conftest.py

import sys
import pytest
import torch
import numpy as np
from config import EnvironmentConfig
from env.environment import HistoricalEnvironment
from agent import MetaSACAgent

# Add project root to sys.path
sys.path.append(".")

@pytest.fixture
def agent():
    """
    Fixture to initialize the MetaSACAgent with mock data.
    """
    config = EnvironmentConfig(
        state_dim=50,
        action_dim=5,
        hidden_dim=128,
        attention_dim=64,
        num_mlp_layers=3,
        dropout_rate=0.1,
        time_encoding_dim=16,
        custom_layers=["KLinePatternLayer", "VolatilityTrackingLayer", "FractalDimensionLayer"],
        window_size=20,
        num_hyperparams=10,
        graph_input_dim=10,
        graph_hidden_dim=32,
        num_graph_layers=2,
        ensemble_size=3,
        weight_decay=1e-5
    )
    mock_data = np.random.randn(2000, config.state_dim).astype(np.float32)  # Increased data points for better simulation
    env = HistoricalEnvironment(mock_data)
    agent = MetaSACAgent(config, env)
    # Populate replay buffer with random data
    for _ in range(config.buffer_capacity // 10):  # Add a fraction to avoid filling it up
        state = env.reset()
        for step in range(10):
            action = np.random.uniform(-1, 1, config.action_dim)
            next_state, reward, done, _ = env.step(action, step)
            agent.replay_buffer.add(state, action, reward, next_state, done, step)
            if done:
                break
            state = next_state
    return agent

@pytest.fixture
def sample_batch():
    """
    Fixture to provide a sample batch of data for testing.
    """
    return {
        'states': np.random.randn(32, 20, 50).astype(np.float32),  # (batch_size, seq_length, state_dim)
        'actions': np.random.randn(32, 5).astype(np.float32),     # (batch_size, action_dim)
        'rewards': np.random.randn(32, 1).astype(np.float32),     # (batch_size, 1)
        'next_states': np.random.randn(32, 20, 50).astype(np.float32),  # (batch_size, seq_length, state_dim)
        'dones': np.random.randint(0, 2, (32, 1)).astype(np.float32),    # (batch_size, 1)
        'time_steps': np.random.randint(0, 1000, (32,)),                 # (batch_size,)
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),  # Example edge index
        'graph_node_features': torch.randn(3, 10)                        # (num_nodes, graph_input_dim)
    }

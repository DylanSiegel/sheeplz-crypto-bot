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
    config = EnvironmentConfig()
    mock_data = np.random.randn(1000, config.state_dim).astype(np.float32)
    env = HistoricalEnvironment(mock_data)
    return MetaSACAgent(config, env)

@pytest.fixture
def sample_batch():
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

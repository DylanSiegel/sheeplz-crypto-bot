# File: tests/test_agent.py

import pytest
import torch
import numpy as np
from config import EnvironmentConfig
from env.environment import HistoricalEnvironment
from agent import MetaSACAgent

@pytest.fixture
def agent_instance():
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
    mock_data = np.random.randn(1000, config.state_dim).astype(np.float32)
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

def test_select_action(agent_instance):
    state = np.random.randn(20, 50).astype(np.float32)  # (seq_length, state_dim)
    time_step = 100
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    graph_node_features = torch.randn(3, 10)  # (num_nodes, graph_input_dim)
    action = agent_instance.select_action(state, time_step, edge_index, graph_node_features, eval=True)
    assert action.shape == (5,), "Action shape mismatch"
    assert np.all(action >= -1) and np.all(action <= 1), "Action values out of bounds"

def test_update_params(agent_instance, sample_batch):
    states = torch.FloatTensor(sample_batch['states']).to(agent_instance.device)
    actions = torch.FloatTensor(sample_batch['actions']).to(agent_instance.device)
    rewards = torch.FloatTensor(sample_batch['rewards']).to(agent_instance.device)
    next_states = torch.FloatTensor(sample_batch['next_states']).to(agent_instance.device)
    dones = torch.FloatTensor(sample_batch['dones']).to(agent_instance.device)
    time_steps = torch.FloatTensor(sample_batch['time_steps']).to(agent_instance.device)

    # Add to replay buffer
    for i in range(states.shape[0]):
        agent_instance.replay_buffer.add(
            states[i].cpu().numpy(),
            actions[i].cpu().numpy(),
            rewards[i].cpu().numpy(),
            next_states[i].cpu().numpy(),
            dones[i].cpu().numpy(),
            int(time_steps[i].item())
        )

    meta_input = np.random.randn(agent_instance.config.batch_size, agent_instance.config.meta_input_dim).astype(np.float32)
    time_memory = list(range(agent_instance.config.window_size))

    losses = agent_instance.update_params_with_training_time_search(
        replay_buffer=agent_instance.replay_buffer,
        meta_input=meta_input,
        time_memory=time_memory,
        update_steps=1
    )

    assert 'actor_loss' in losses, "Missing actor_loss in losses"
    assert 'critic1_loss' in losses, "Missing critic1_loss in losses"
    assert 'critic2_loss' in losses, "Missing critic2_loss in losses"
    assert 'meta_loss' in losses, "Missing meta_loss in losses"
    assert 'distiller_loss' in losses, "Missing distiller_loss in losses"
    assert 'market_mode_loss' in losses, "Missing market_mode_loss in losses"
    assert 'high_level_loss' in losses, "Missing high_level_loss in losses"
    assert 'alpha' in losses, "Missing alpha in losses"

    # Additional Assertions
    assert isinstance(losses['actor_loss'], float), "actor_loss should be a float"
    assert isinstance(losses['critic1_loss'], float), "critic1_loss should be a float"
    assert isinstance(losses['critic2_loss'], float), "critic2_loss should be a float"
    assert isinstance(losses['meta_loss'], float), "meta_loss should be a float"
    assert isinstance(losses['distiller_loss'], float), "distiller_loss should be a float"
    assert isinstance(losses['market_mode_loss'], float), "market_mode_loss should be a float"
    assert isinstance(losses['high_level_loss'], float), "high_level_loss should be a float"
    assert isinstance(losses['alpha'], float), "alpha should be a float"

def test_model_save_load(agent_instance, tmp_path):
    """
    Test saving and loading of the agent model.
    """
    save_path = tmp_path / "metasac_test.pth"
    agent_instance.save(str(save_path))

    # Create a new agent instance and load the saved state
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
    mock_data = np.random.randn(1000, config.state_dim).astype(np.float32)
    env = HistoricalEnvironment(mock_data)
    new_agent = MetaSACAgent(config, env)

    # Load the saved state
    new_agent.load(str(save_path))

    # Verify that parameters match
    for param1, param2 in zip(agent_instance.parameters(), new_agent.parameters()):
        assert torch.allclose(param1, param2), "Model parameters do not match after loading."

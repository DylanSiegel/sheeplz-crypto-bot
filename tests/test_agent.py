# File: test_agent.py

import pytest
import torch
import numpy as np
from config import EnvironmentConfig
from env.environment import HistoricalEnvironment
from agent import MetaSACAgent

@pytest.fixture
def agent_instance():
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
        ensemble_size=3
    )
    mock_data = np.random.randn(1000, config.state_dim).astype(np.float32)
    env = HistoricalEnvironment(mock_data)
    agent = MetaSACAgent(config, env)
    return agent

def test_select_action(agent_instance):
    state = np.random.randn(20, 50).astype(np.float32)  # (seq_length, state_dim)
    time_step = 100
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    graph_node_features = torch.randn(3, 10)  # (num_nodes, graph_input_dim)
    action = agent_instance.select_action(state, time_step, edge_index, graph_node_features, eval=True)
    assert action.shape == (5,), "Action shape mismatch"

def test_update_params(agent_instance, sample_batch):
    states = torch.FloatTensor(sample_batch['states']).to(agent_instance.device)
    actions = torch.FloatTensor(sample_batch['actions']).to(agent_instance.device)
    rewards = torch.FloatTensor(sample_batch['rewards']).to(agent_instance.device)
    next_states = torch.FloatTensor(sample_batch['next_states']).to(agent_instance.device)
    dones = torch.FloatTensor(sample_batch['dones']).to(agent_instance.device)
    time_steps = torch.FloatTensor(sample_batch['time_steps']).to(agent_instance.device)
    edge_index = sample_batch['edge_index'].to(agent_instance.device)
    graph_node_features = sample_batch['graph_node_features'].to(agent_instance.device)

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

# File: tests/test_agent.py

import pytest
import torch
import numpy as np
from config import EnvironmentConfig
from env.environment import HistoricalEnvironment
from agent import MetaSACAgent
from networks import (
    KLinePatternLayer,
    VolatilityTrackingLayer,
    FractalDimensionLayer,
    APELU,
    MomentumActivation,
    VolatilityAdaptiveActivation,
    TransformerEncoderLayerCustom,
    TransformerEncoderCustom,
    MultiHeadAttentionCustom,
    BaseMLP,
    AdaptiveModulationMLP,
    SinusoidalTimeEncoding,
    TimeAwareBias,
    PolicyDistillerEnsemble,
    HighLevelPolicy,
    MarketModeClassifier,
    MetaController,
)
from replay_buffer import ReplayBuffer
from reward import calculate_reward

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

# Example unit test for KLinePatternLayer
def test_kline_pattern_layer():
    layer = KLinePatternLayer(hidden_dim=32)
    # Create a batch of 2 sequences, each 5 time steps long, with 4 features (OHLC)
    x = torch.randn(2, 5, 4)

    # Simulate a bullish engulfing pattern in the first sequence
    x[0, 0, :] = torch.tensor([10, 12, 8, 9])  # Prev: O=10, H=12, L=8, C=9 (bearish)
    x[0, 1, :] = torch.tensor([8, 13, 7, 12])  # Curr: O=8, H=13, L=7, C=12 (bullish engulfing)

    # Simulate a bearish engulfing pattern in the second sequence
    x[1, 2, :] = torch.tensor([15, 17, 14, 16]) # Prev: O=15, H=17, L=14, C=16 (bullish)
    x[1, 3, :] = torch.tensor([17, 18, 15, 14]) # Curr: O=17, H=18, L=15, C=14 (bearish engulfing)

    patterns = layer.detect_patterns(x)

    # Check if patterns are detected correctly
    assert patterns[0, 1, 0] == 1  # Bullish engulfing at time step 1 in sequence 0
    assert patterns[0, 1, 1] == 0  # Not bearish engulfing
    assert patterns[0, 1, 2] == 0  # Not doji
    assert patterns[0, 1, 3] == 0  # Not hammer
    assert patterns[0, 1, 4] == 0  # Not inverted hammer
    assert patterns[1, 3, 1] == 1  # Bearish engulfing at time step 3 in sequence 1
    assert patterns[1, 3, 0] == 0  # Not bullish engulfing
    assert patterns[1, 3, 2] == 0  # Not doji
    assert patterns[1, 3, 3] == 0  # Not hammer
    assert patterns[1, 3, 4] == 0  # Not inverted hammer
    # Add more assertions for other patterns and sequences as needed

# Example unit test for VolatilityTrackingLayer
def test_volatility_tracking_layer():
    layer = VolatilityTrackingLayer(hidden_dim=32, window_size=3)
    # Create a batch of 1 sequence, 5 time steps long, with 4 features (OHLC)
    x = torch.tensor([
        [[10, 11, 9, 10.5]],  # Assume close price is the last feature
        [[10.5, 12, 10, 11]],
        [[11, 11.5, 10.5, 11.2]],
        [[11.2, 13, 11, 12.5]],
        [[12.5, 13.5, 12, 13]]
    ], dtype=torch.float32)

    volatility_measures = layer.calculate_volatility_measures(x[:, :, 3].unsqueeze(-1), x)

    # Manually calculate std_dev for a window and compare
    window = x[0, 1:4, 3]  # Time steps 1, 2, 3
    log_returns = torch.log(window[1:] / window[:-1])
    expected_std_dev = torch.std(log_returns).item()
    calculated_std_dev = volatility_measures[0, 3, 0].item()

    assert pytest.approx(calculated_std_dev, abs=1e-5) == expected_std_dev

# Example unit test for APELU activation function
def test_apelu_activation():
    activation = APELU(alpha_init=0.01, beta_init=1.0)
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected_output = torch.where(x >= 0, x, 0.01 * x * torch.exp(x))

    output = activation(x)
    assert torch.allclose(output, expected_output)

# Example integration test for updating critics
def test_update_critics(agent_instance, sample_batch):
    states = torch.FloatTensor(sample_batch['states']).to(agent_instance.device)
    actions = torch.FloatTensor(sample_batch['actions']).to(agent_instance.device)
    rewards = torch.FloatTensor(sample_batch['rewards']).to(agent_instance.device)
    next_states = torch.FloatTensor(sample_batch['next_states']).to(agent_instance.device)
    dones = torch.FloatTensor(sample_batch['dones']).to(agent_instance.device)
    time_steps = torch.FloatTensor(sample_batch['time_steps']).to(agent_instance.device)

    # Get initial parameters
    initial_critic1_params = [p.clone() for p in agent_instance.critic1.parameters()]
    initial_critic2_params = [p.clone() for p in agent_instance.critic2.parameters()]

    # Mock Q-targets
    q_targets = torch.randn_like(rewards)

    critic1_loss, critic2_loss = agent_instance.update_critics(states, actions, time_steps, q_targets)

    # Check if parameters have been updated
    for initial_param, updated_param in zip(initial_critic1_params, agent_instance.critic1.parameters()):
        assert not torch.equal(initial_param, updated_param), "Critic1 parameters did not update"

    for initial_param, updated_param in zip(initial_critic2_params, agent_instance.critic2.parameters()):
        assert not torch.equal(initial_param, updated_param), "Critic2 parameters did not update"

    # Check if losses are valid floats
    assert isinstance(critic1_loss, float), "Critic1 loss is not a float"
    assert isinstance(critic2_loss, float), "Critic2 loss is not a float"

# Unit test for MomentumActivation
def test_momentum_activation():
    activation = MomentumActivation(momentum_sensitivity=0.5)
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float32)
    expected_output = x * (1 + 0.5 * torch.tanh(x))
    output = activation(x)
    assert torch.allclose(output, expected_output, atol=1e-6), "MomentumActivation output mismatch"

# Unit test for VolatilityAdaptiveActivation
def test_volatility_adaptive_activation():
    activation = VolatilityAdaptiveActivation(initial_scale=0.5)
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    volatility = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float32)
    expected_output = x * (1 + 0.5 * torch.tanh(volatility))
    output = activation(x, volatility)
    assert torch.allclose(output, expected_output, atol=1e-6), "VolatilityAdaptiveActivation output mismatch"

# Unit test for TransformerEncoderLayerCustom
def test_transformer_encoder_layer_custom():
    layer = TransformerEncoderLayerCustom(embed_dim=32, num_heads=4, dim_feedforward=64, dropout=0.1)
    x = torch.randn(10, 5, 32)  # (seq_length, batch_size, embed_dim)
    output = layer(x)
    assert output.shape == x.shape, "TransformerEncoderLayerCustom output shape mismatch"

# Unit test for TransformerEncoderCustom
def test_transformer_encoder_custom():
    encoder = TransformerEncoderCustom(embed_dim=32, num_heads=4, num_layers=2, dim_feedforward=64, dropout=0.1)
    x = torch.randn(10, 5, 32)  # (seq_length, batch_size, embed_dim)
    output = encoder(x)
    assert output.shape == x.shape, "TransformerEncoderCustom output shape mismatch"

# Unit test for MultiHeadAttentionCustom
def test_multihead_attention_custom():
    attention = MultiHeadAttentionCustom(embed_dim=32, num_heads=4, dropout=0.1)
    x = torch.randn(10, 5, 32)  # (seq_length, batch_size, embed_dim)
    output = attention(x)
    assert output.shape == x.shape, "MultiHeadAttentionCustom output shape mismatch"

# Unit test for BaseMLP
def test_base_mlp():
    mlp = BaseMLP(input_dim=50, hidden_dim=64, output_dim=5, num_layers=3, dropout_rate=0.1, use_custom_layers=False, window_size=20)
    x = torch.randn(32, 50)  # (batch_size, input_dim)
    output = mlp(x)
    assert output.shape == (32, 5), "BaseMLP output shape mismatch"

# Unit test for AdaptiveModulationMLP
def test_adaptive_modulation_mlp():
    mlp = AdaptiveModulationMLP(
        input_dim=50, hidden_dim=64, output_dim=5, num_layers=3, dropout_rate=0.1,
        time_encoding_dim=16, use_custom_layers=False, window_size=20
    )
    x = torch.randn(32, 50)  # (batch_size, input_dim)
    time_step = torch.randint(0, 100, (32,))  # (batch_size,)
    output = mlp(x, time_step)
    assert output.shape == (32, 5), "AdaptiveModulationMLP output shape mismatch"

# Unit test for SinusoidalTimeEncoding
def test_sinusoidal_time_encoding():
    encoding = SinusoidalTimeEncoding(time_encoding_dim=16)
    time_step = torch.tensor([0, 1, 2, 3], dtype=torch.float32)  # (batch_size,)
    output = encoding(time_step)
    assert output.shape == (4, 16), "SinusoidalTimeEncoding output shape mismatch"

# Unit test for TimeAwareBias
def test_time_aware_bias():
    bias = TimeAwareBias(input_dim=64, time_encoding_dim=16, hidden_dim=32)
    time_encoding = torch.randn(32, 16)  # (batch_size, time_encoding_dim)
    output = bias(time_encoding)
    assert output.shape == (32, 64), "TimeAwareBias output shape mismatch"

# Unit test for PolicyDistillerEnsemble
def test_policy_distiller_ensemble(agent_instance):
    ensemble = PolicyDistillerEnsemble(agent_instance.specialist_policies, agent_instance.config)
    state = torch.randn(32, 20, 50)  # (batch_size, seq_length, state_dim)
    time_step = torch.randint(0, 100, (32,))  # (batch_size,)
    mu, log_sigma = ensemble(state, time_step)
    assert mu.shape == (32, 5), "PolicyDistillerEnsemble mu output shape mismatch"
    assert log_sigma.shape == (32, 5), "PolicyDistillerEnsemble log_sigma output shape mismatch"

# Unit test for HighLevelPolicy
def test_high_level_policy():
    policy = HighLevelPolicy(state_dim=50, hidden_dim=64)
    state = torch.randn(32, 50)  # (batch_size, state_dim)
    output = policy(state)
    assert output.shape == (32, 1), "HighLevelPolicy output shape mismatch"

# Unit test for MarketModeClassifier
def test_market_mode_classifier():
    classifier = MarketModeClassifier(input_dim=50, hidden_dim=64, output_dim=3)
    state = torch.randn(32, 50)  # (batch_size, state_dim)
    output = classifier(state)
    assert output.shape == (32, 3), "MarketModeClassifier output shape mismatch"

# Unit test for MetaController
def test_meta_controller(agent_instance):
    meta_controller = MetaController(config=agent_instance.config)
    state = torch.randn(32, 50)
    reward_stats = torch.randn(32, 2)
    output = meta_controller(state, reward_stats)
    assert output.shape == (13,), "MetaController output shape mismatch"

# Unit test for FractalDimensionLayer
def test_fractal_dimension_layer():
    layer = FractalDimensionLayer(hidden_dim=32, max_k=5, buffer_size=50)
    # Create a batch of 2 sequences, each 10 time steps long, with 4 features (OHLC)
    x = torch.randn(2, 10, 4)

    # Simulate a specific pattern in the first sequence
    x[0, :, 3] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])  # Example increasing sequence

    output = layer(x)
    assert output.shape == (2, 10, 32), "FractalDimensionLayer output shape mismatch"

    # Add more specific assertions based on the expected fractal dimension of your simulated data
    # This requires calculating the expected fractal dimension using another method for comparison
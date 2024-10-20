# src/rl_agent/custom_policy.py

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import torch.nn.functional as F
import gym

class LiquidNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using Liquid Neural Network architecture.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=128):
        super(LiquidNNFeaturesExtractor, self).__init__(observation_space, features_dim)
        input_size = observation_space.shape[0]
        sequence_length = 30  # Should match the sequence length used in data preparation
        
        # Define LNN architecture
        self.gru = nn.GRU(input_size, hidden_size=64, batch_first=True)
        self.attention = nn.Linear(64, 1)
        self.fc = nn.Linear(64, features_dim)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: [batch_size, sequence_length, input_size]
        gru_out, _ = self.gru(observations)
        attn_weights = F.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        features = F.relu(self.fc(context))
        return features

class LiquidNNPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy with Liquid Neural Network Feature Extractor.
    """
    def __init__(self, *args, **kwargs):
        super(LiquidNNPolicy, self).__init__(*args, **kwargs,
                                            features_extractor_class=LiquidNNFeaturesExtractor,
                                            features_extractor_kwargs={'features_dim': 128})
        # Optionally, customize the actor and critic networks further here

    def _build_mlp_extractor(self):
        # Override to use custom feature extractor
        pass

# src/agents/agent_factory.py

from typing import Dict, Any
from .base_agent import BaseAgent
from .dqn_agent import DQNAgent
from ..models.base_model import BaseModel
from ..utils.config_manager import ConfigManager

class AgentFactory:
    @staticmethod
    def create(config: ConfigManager, model: BaseModel) -> BaseAgent:
        agent_type = config.get('agent.type')
        agent_params = config.get('agent.params')

        if agent_type == 'DQN':
            return DQNAgent(model, **agent_params)
        # Add more agent types as needed
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
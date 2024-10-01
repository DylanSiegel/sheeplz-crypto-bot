# File: src/models/base_model.py

import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for trading models.
    """
    @abstractmethod
    def get_action(self, state):
        """
        Determines the action to take based on the current state.
        """
        pass

    @abstractmethod
    def update(self, optimizer, criterion, state, action, reward, next_state, done):
        """
        Updates the model parameters based on the transition.
        """
        pass

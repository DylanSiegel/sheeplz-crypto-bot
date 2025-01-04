# replay_buffer.py
import random
import numpy as np
from typing import Tuple, List

class ReplayBuffer:
    """Stores transitions for training."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done, time_step):
        """Adds transition to buffer."""
        transition = (state, action, reward, next_state, done, time_step)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, None, None]:
        """Samples a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        return batch, None, None

    def __len__(self):
        return len(self.buffer)
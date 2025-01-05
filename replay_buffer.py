# File: replay_buffer.py

import random
from typing import List, Tuple

class ReplayBuffer:
    """Stores transitions for training."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position: int = 0

    def add(self, state, action, reward, next_state, done, time_step):
        """Adds a transition to the buffer."""
        transition = (state, action, reward, next_state, done, time_step)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple]:
        """Samples a batch of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

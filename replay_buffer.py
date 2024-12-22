import numpy as np
from typing import Tuple, List

class ReplayBuffer:
    """Experience replay buffer with prioritized sampling."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.epsilon = 1e-6  # Small constant to prevent zero priorities

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max(self.priorities.max(), self.epsilon)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in buffer")

        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities + self.epsilon

    def __len__(self) -> int:
        return len(self.buffer)

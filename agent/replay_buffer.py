"""Fixed-size circular buffer of (state, action, reward, next_state, done)."""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, state_dim, seed=0):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0          # where the next experience is written
        self.size = 0         # how many experiences are stored
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def push(self, state, action, reward, next_state, done):
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity      # overwrite oldest when full
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} from a buffer holding {self.size}")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )
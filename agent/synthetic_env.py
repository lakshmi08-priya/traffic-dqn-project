"""Toy two-queue environment with the same interface as TrafficEnv.

State (4 numbers): [queue_0, queue_1, green_is_0, green_is_1], queues in [0, 1].
Actions: 0 = KEEP, 1 = SWITCH. Switching wastes the step (like yellow time).
Reward: minus the total queue length.
"""
import numpy as np

KEEP, SWITCH = 0, 1


class SyntheticTrafficEnv:
    state_dim = 4
    n_actions = 2

    def __init__(self, episode_len=200, seed=0, discharge=0.10, max_arrival=0.08):
        self.episode_len = episode_len
        self.discharge = discharge
        self.max_arrival = max_arrival
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.queues = np.array([0.3, 0.3])
        self.green = 0
        self.t = 0
        return self.get_state()

    def get_state(self):
        onehot = [1.0 if self.green == i else 0.0 for i in range(2)]
        return np.array(list(self.queues) + onehot, dtype=np.float32)

    def step(self, action):
        if action not in (KEEP, SWITCH):
            raise ValueError(f"Invalid action {action!r}")
        if action == SWITCH:
            self.green = 1 - self.green           # no discharge this step
        else:
            self.queues[self.green] = max(0.0, self.queues[self.green] - self.discharge)
        self.queues = np.clip(self.queues + self.rng.uniform(0.0, self.max_arrival, size=2), 0.0, 1.0)
        self.t += 1
        reward = -float(self.queues.sum())
        done = self.t >= self.episode_len
        return self.get_state(), reward, done, {}
    
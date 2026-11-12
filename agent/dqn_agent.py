"""DQN agent: epsilon-greedy, replay, target network, Huber loss, checkpoints."""
import copy
import json
import os

import numpy as np
import torch
from torch import nn

from agent.q_network import QNetwork
from agent.replay_buffer import ReplayBuffer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "dqn_config.json")


def load_dqn_config(path=None, overrides=None):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = json.load(f)
    if overrides:
        cfg.update(overrides)
    return cfg


class DQNAgent:
    def __init__(self, state_dim, n_actions, config=None, seed=0):
        self.cfg = config or load_dqn_config()
        self.state_dim, self.n_actions = state_dim, n_actions
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.online = QNetwork(state_dim, n_actions, self.cfg["hidden"])
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)                 # target is never trained directly
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg["lr"])
        self.loss_fn = nn.SmoothL1Loss()            # Huber loss
        self.buffer = ReplayBuffer(self.cfg["buffer_size"], state_dim, seed)

        self.epsilon = self.cfg["epsilon_start"]
        self.learn_steps = 0

    # ---------------------------------------------------------- acting
    def select_action(self, state, greedy=False):
        """Epsilon-greedy. With greedy=True, always pick the best-looking action."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(self.cfg["epsilon_end"], self.epsilon * self.cfg["epsilon_decay"])

    # --------------------------------------------------------- learning
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def compute_targets(self, rewards, next_states, dones):
        """Bellman target: r + gamma * max_a' Q_target(s', a') * (1 - done)."""
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            return rewards + self.cfg["gamma"] * next_q * (1.0 - dones)

    def learn(self):
        """One gradient step on a random batch. Returns the loss, or None if too little data."""
        if len(self.buffer) < max(self.cfg["batch_size"], self.cfg["learning_starts"]):
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg["batch_size"])
        q_taken = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = self.compute_targets(rewards, next_states, dones)
        loss = self.loss_fn(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg["grad_clip"])
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.cfg["target_update_steps"] == 0:
            self.update_target()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------ checkpoints
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "config": self.cfg,
        }, path)

    @classmethod
    def load(cls, path, seed=0):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(ckpt["state_dim"], ckpt["n_actions"], ckpt["config"], seed)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.epsilon = ckpt["epsilon"]
        agent.learn_steps = ckpt["learn_steps"]
        return agent
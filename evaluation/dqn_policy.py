"""Greedy policy backed by a saved DQN checkpoint, usable by run_episode."""
from agent.dqn_agent import DQNAgent


class DQNPolicy:
    def __init__(self, checkpoint_path):
        self.agent = DQNAgent.load(checkpoint_path)

    def act(self, state, info):
        return self.agent.select_action(state, greedy=True)
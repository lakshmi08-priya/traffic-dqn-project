"""Train the DQN on the toy environment and compare with simple policies."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.dqn_agent import DQNAgent  # noqa: E402
from agent.synthetic_env import SyntheticTrafficEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=150)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


def evaluate(policy, episodes=20, seed_base=10_000):
    """Average return over fresh toy episodes (same seeds for every policy)."""
    returns = []
    for i in range(episodes):
        env = SyntheticTrafficEnv(seed=seed_base + i)
        state, done, total = env.reset(), False, 0.0
        while not done:
            state, reward, done, _ = env.step(policy(state))
            total += reward
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


env = SyntheticTrafficEnv(seed=args.seed)
agent = DQNAgent(env.state_dim, env.n_actions, seed=args.seed)

for ep in range(1, args.episodes + 1):
    state, done, ep_return, losses = env.reset(), False, 0.0, []
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        # Episode ends by time limit, not by a real terminal event, so store done=False.
        agent.store(state, action, reward, next_state, False)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        state, ep_return = next_state, ep_return + reward
    agent.decay_epsilon()
    if ep % 10 == 0 or ep == 1:
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"episode {ep:3d}  return={ep_return:8.2f}  epsilon={agent.epsilon:.3f}  loss={mean_loss:.4f}")

agent.save("models/synthetic_dqn.pt")

rng = np.random.default_rng(0)
results = {
    "random": evaluate(lambda s: int(rng.integers(2))),
    "always KEEP": evaluate(lambda s: 0),
    "DQN (greedy)": evaluate(lambda s: agent.select_action(s, greedy=True)),
}
print("\n--- mean return over 20 unseen toy episodes (higher is better) ---")
for name, (mean, std) in results.items():
    print(f"{name:<14} {mean:8.2f} +/- {std:.2f}")
print("\nSaved models/synthetic_dqn.pt")
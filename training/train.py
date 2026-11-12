"""Train the DQN on the SUMO environment. Saves checkpoints and per-episode logs."""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.dqn_agent import DQNAgent, load_dqn_config  # noqa: E402
from environment.traffic_env import TrafficEnv  # noqa: E402

SCENARIOS = ["low", "medium", "high", "uneven"]
TRAIN_SEED_BASE = 100   # training seeds are 100+; evaluation uses 0-4 (unseen)


def train(episodes, episode_seconds, out_path, log_path, seed=0, dqn_overrides=None):
    env = TrafficEnv(overrides={"episode_seconds": episode_seconds})
    agent = DQNAgent(env.state_dim, env.n_actions,
                     load_dqn_config(overrides=dqn_overrides), seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(1, episodes + 1):
        scenario = SCENARIOS[(ep - 1) % len(SCENARIOS)]
        route_seed = TRAIN_SEED_BASE + int(rng.integers(0, 50))
        state = env.reset(scenario, route_seed)
        done, ep_return, losses, steps, switches = False, 0.0, [], 0, 0
        t0 = time.time()
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, False)   # time limit, not terminal
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state, ep_return = next_state, ep_return + reward
            steps += 1
            switches += info["switched"]
        agent.decay_epsilon()
        row = {"episode": ep, "scenario": scenario, "route_seed": route_seed,
               "return": ep_return, "epsilon": agent.epsilon,
               "loss": float(np.mean(losses)) if losses else float("nan"),
               "steps": steps, "switches": switches,
               "arrived": info["arrived_total"], "safety_violations": info["safety_violations"],
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"ep {ep:3d} {scenario:<6} return={ep_return:8.2f} eps={agent.epsilon:.3f} "
              f"loss={row['loss']:.4f} switches={switches} arrived={row['arrived']} ({row['seconds']}s)")
        agent.save(out_path)
        pd.DataFrame(rows).to_csv(log_path, index=False)
    env.close()
    return agent


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--episode-seconds", type=int, default=1800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/dqn_sumo.pt")
    p.add_argument("--log", default="results/csv/training_log.csv")
    a = p.parse_args()
    os.makedirs("results/csv", exist_ok=True)
    train(a.episodes, a.episode_seconds, a.out, a.log, a.seed)
    print(f"\nSaved {a.out} and {a.log}")
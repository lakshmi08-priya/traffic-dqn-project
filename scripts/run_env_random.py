"""Run one episode with a simple policy to check the environment end to end."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.traffic_env import TrafficEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", default="medium")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--policy", choices=["random", "keep"], default="random")
parser.add_argument("--episode-seconds", type=int, default=600)
args = parser.parse_args()

env = TrafficEnv(overrides={"episode_seconds": args.episode_seconds})
rng = np.random.default_rng(args.seed)
state = env.reset(args.scenario, args.seed)
print("state dim:", state.shape, " min/max:", float(state.min()), float(state.max()))

total_reward, steps, switches, rejected = 0.0, 0, 0, 0
done = False
while not done:
    action = int(rng.integers(2)) if args.policy == "random" else 0
    state, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1
    switches += info["switched"]
    rejected += info["rejected"]
    if steps % 20 == 0:
        print(f"t={info['sim_time']:.0f}s green={info['green_index']} wait={info['total_wait']:.0f} "
              f"queue={info['mean_queue']:.1f} reward={reward:+.2f}")
env.close()

print("\n--- summary ---")
print(f"policy={args.policy} scenario={args.scenario} seed={args.seed}")
print(f"steps={steps} sim_time={info['sim_time']:.0f}s switches={switches} rejected={rejected}")
print(f"total reward={total_reward:.2f} vehicles arrived={info['arrived_total']}")
print(f"safety violations={info['safety_violations']}")

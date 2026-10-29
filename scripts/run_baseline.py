"""Run the fixed-time baseline on all scenarios and seeds; save results to CSV."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.traffic_env import TrafficEnv  # noqa: E402
from evaluation.fixed_time import FixedTimePolicy, run_episode  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", nargs="+", default=["low", "medium", "high"])
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--green-time", type=int, default=30)
parser.add_argument("--episode-seconds", type=int, default=3600)
args = parser.parse_args()

env = TrafficEnv(overrides={"episode_seconds": args.episode_seconds})
policy = FixedTimePolicy(args.green_time)
rows = []
for scenario in args.scenarios:
    for seed in range(args.seeds):
        m = run_episode(policy, scenario, seed, env=env)
        m.update(controller="fixed_time", scenario=scenario, seed=seed,
                 green_time=args.green_time, episode_seconds=args.episode_seconds)
        rows.append(m)
        print(f"{scenario} seed{seed}: wait={m['mean_wait']:.1f}s travel={m['mean_travel_time']:.1f}s "
              f"queue={m['mean_queue']:.1f} throughput={m['throughput']} left={m['vehicles_left']}")

df = pd.DataFrame(rows)
os.makedirs("results/csv", exist_ok=True)
df.to_csv("results/csv/baseline_fixed_time.csv", index=False)

print("\n--- mean over seeds ---")
print(df.groupby("scenario")[["mean_wait", "mean_travel_time", "mean_queue", "throughput"]]
        .agg(["mean", "std"]).round(2))
print("\nSaved results/csv/baseline_fixed_time.csv")
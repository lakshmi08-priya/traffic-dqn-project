"""Fixed-time controller and a generic episode runner used for any policy."""
import os
import tempfile

from environment.traffic_env import KEEP, SWITCH, TrafficEnv
from evaluation.metrics import compute_metrics, read_tripinfo


class FixedTimePolicy:
    """SWITCH once the current green has lasted `green_time` seconds, else KEEP."""

    def __init__(self, green_time=30):
        self.green_time = green_time

    def act(self, info):
        return SWITCH if info["green_elapsed"] >= self.green_time else KEEP


def run_episode(policy, scenario, seed, env=None, episode_seconds=3600):
    """Run one full episode with `policy` (needs .act(info)) and return a metrics dict."""
    env = env or TrafficEnv(overrides={"episode_seconds": episode_seconds})
    with tempfile.TemporaryDirectory() as tmp:
        trip_path = os.path.join(tmp, "tripinfo.xml")
        env.reset(scenario, seed, tripinfo_path=trip_path)
        info = {"green_elapsed": 0}
        queue_samples, done = [], False
        while not done:
            _, _, done, info = env.step(policy.act(info))
            queue_samples.append(info["mean_queue"])
        left = env.vehicles_in_network()
        env.close()                      # tripinfo is flushed on close
        waits, durs = read_tripinfo(trip_path)
    return compute_metrics(waits, durs, queue_samples, left)
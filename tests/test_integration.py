"""Short end-to-end check: train briefly on SUMO, save, load, evaluate."""
from evaluation.dqn_policy import DQNPolicy
from evaluation.fixed_time import FixedTimePolicy, run_episode
from training.train import train


def test_train_save_load_evaluate(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    log = str(tmp_path / "log.csv")
    train(episodes=2, episode_seconds=120, out_path=ckpt, log_path=log, seed=0,
          dqn_overrides={"learning_starts": 20, "batch_size": 16})
    policy = DQNPolicy(ckpt)
    m = run_episode(policy, "low", 0, episode_seconds=120)
    assert m["throughput"] >= 0 and m["mean_queue"] >= 0
    base = run_episode(FixedTimePolicy(30), "low", 0, episode_seconds=120)
    assert base["throughput"] >= 0
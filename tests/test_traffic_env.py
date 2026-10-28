import numpy as np
import pytest

from environment.traffic_env import KEEP, SWITCH, TrafficEnv


@pytest.fixture
def env():
    e = TrafficEnv(overrides={"episode_seconds": 120})
    e.reset("low", 0)
    yield e
    e.close()


def test_state_shape_and_range(env):
    s = env.get_state()
    assert s.shape == (env.state_dim,)
    assert s.dtype == np.float32
    assert np.all(np.isfinite(s))
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_invalid_action_raises(env):
    with pytest.raises(ValueError):
        env.step(2)


def test_step_returns_expected_types(env):
    s, r, done, info = env.step(KEEP)
    assert s.shape == (env.state_dim,)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert "sim_time" in info and "total_wait" in info


def test_keep_stays_in_same_green(env):
    _, _, _, info = env.step(KEEP)
    assert info["green_index"] == 0
    assert not info["switched"]


def test_early_switch_is_rejected(env):
    _, _, _, info = env.step(SWITCH)          # green has lasted 0 s < min_green
    assert info["rejected"] and not info["switched"]
    assert info["green_index"] == 0


def test_switch_after_min_green_changes_green(env):
    env.step(KEEP)
    env.step(KEEP)                            # 10 s of green
    _, _, _, info = env.step(SWITCH)
    assert info["switched"] and not info["rejected"]
    assert info["green_index"] == 1
    assert info["safety_violations"] == 0


def test_forced_switch_after_max_green():
    e = TrafficEnv(overrides={"episode_seconds": 200, "max_green": 20})
    e.reset("low", 0)
    for _ in range(4):                        # 20 s of green
        e.step(KEEP)
    _, _, _, info = e.step(KEEP)
    e.close()
    assert info["forced"] and info["switched"]


def test_episode_terminates(env):
    done, steps = False, 0
    while not done and steps < 200:
        _, _, done, info = env.step(KEEP)
        steps += 1
    assert done
    assert info["sim_time"] >= 120


def test_no_conflicting_greens_with_random_actions(env):
    rng = np.random.default_rng(0)
    done = False
    while not done:
        _, _, done, info = env.step(int(rng.integers(2)))
    assert info["safety_violations"] == 0
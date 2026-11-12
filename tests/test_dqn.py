import numpy as np
import pytest
import torch
from torch import nn

from agent.dqn_agent import DQNAgent, load_dqn_config
from agent.q_network import QNetwork
from agent.replay_buffer import ReplayBuffer
from agent.synthetic_env import SyntheticTrafficEnv


def make_agent(**overrides):
    cfg = load_dqn_config(overrides=overrides)
    return DQNAgent(4, 2, cfg, seed=0)


def fill(agent, n):
    rng = np.random.default_rng(1)
    for _ in range(n):
        agent.store(rng.random(4), int(rng.integers(2)), float(rng.normal()), rng.random(4), False)


def test_qnetwork_output_shape():
    out = QNetwork(27, 2, 64)(torch.rand(5, 27))
    assert out.shape == (5, 2)


def test_buffer_push_and_sample_shapes():
    buf = ReplayBuffer(100, 4)
    for i in range(10):
        buf.push(np.ones(4) * i, i % 2, float(i), np.ones(4) * (i + 1), i == 9)
    s, a, r, s2, d = buf.sample(6)
    assert s.shape == (6, 4) and s2.shape == (6, 4)
    assert a.shape == (6,) and a.dtype == torch.int64
    assert r.shape == (6,) and d.shape == (6,)


def test_buffer_overwrites_oldest_when_full():
    buf = ReplayBuffer(5, 4)
    for i in range(8):
        buf.push(np.zeros(4), 0, float(i), np.zeros(4), False)
    assert len(buf) == 5
    assert set(buf.rewards.tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}


def test_buffer_sample_too_large_raises():
    buf = ReplayBuffer(10, 4)
    buf.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
    with pytest.raises(ValueError):
        buf.sample(5)


def test_greedy_action_matches_argmax():
    agent = make_agent()
    state = np.random.default_rng(0).random(4).astype(np.float32)
    with torch.no_grad():
        expected = int(agent.online(torch.from_numpy(state).unsqueeze(0)).argmax())
    agent.epsilon = 0.0
    assert agent.select_action(state) == expected
    agent.epsilon = 1.0
    assert agent.select_action(state, greedy=True) == expected


def test_epsilon_one_explores_both_actions():
    agent = make_agent()
    agent.epsilon = 1.0
    actions = {agent.select_action(np.zeros(4, dtype=np.float32)) for _ in range(100)}
    assert actions == {0, 1}


def test_bellman_target_matches_hand_calculation():
    agent = make_agent(gamma=0.9)

    class FixedQ(nn.Module):                    # always returns Q(KEEP)=1.0, Q(SWITCH)=2.0
        def forward(self, x):
            return torch.tensor([[1.0, 2.0]]).repeat(len(x), 1)

    agent.target = FixedQ()
    rewards = torch.tensor([0.3, 0.3])
    dones = torch.tensor([0.0, 1.0])
    targets = agent.compute_targets(rewards, torch.zeros(2, 4), dones)
    assert targets[0].item() == pytest.approx(0.3 + 0.9 * 2.0)   # 2.1
    assert targets[1].item() == pytest.approx(0.3)               # done: no future term


def test_learn_returns_none_with_too_little_data():
    agent = make_agent(learning_starts=50)
    fill(agent, 10)
    assert agent.learn() is None


def test_loss_decreases_on_fixed_data():
    agent = make_agent(batch_size=64, learning_starts=64, target_update_steps=10**9, lr=0.01)
    fill(agent, 64)
    first = agent.learn()
    for _ in range(300):
        last = agent.learn()
    assert last < first


def test_target_network_frozen_between_updates():
    agent = make_agent(batch_size=32, learning_starts=32, target_update_steps=10**9)
    fill(agent, 64)
    before = [p.clone() for p in agent.target.parameters()]
    for _ in range(20):
        agent.learn()
    for b, p in zip(before, agent.target.parameters()):
        assert torch.equal(b, p)


def test_update_target_copies_online_weights():
    agent = make_agent(batch_size=32, learning_starts=32, target_update_steps=10**9)
    fill(agent, 64)
    for _ in range(20):
        agent.learn()
    agent.update_target()
    for po, pt in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(po, pt)


def test_epsilon_decays_to_minimum_and_stops():
    agent = make_agent(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.5)
    for _ in range(20):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.05)


def test_save_and_load_roundtrip(tmp_path):
    agent = make_agent()
    agent.epsilon = 0.37
    path = str(tmp_path / "ckpt.pt")
    agent.save(path)
    loaded = DQNAgent.load(path)
    x = torch.rand(3, 4)
    with torch.no_grad():
        assert torch.allclose(agent.online(x), loaded.online(x))
    assert loaded.epsilon == pytest.approx(0.37)
    assert loaded.state_dim == 4 and loaded.n_actions == 2


def test_synthetic_env_reset_step_and_termination():
    env = SyntheticTrafficEnv(episode_len=10, seed=0)
    s = env.reset()
    assert s.shape == (4,) and s.dtype == np.float32
    done, steps = False, 0
    while not done:
        s, r, done, _ = env.step(0)
        assert s.shape == (4,) and r <= 0.0
        steps += 1
    assert steps == 10


def test_synthetic_env_invalid_action_raises():
    env = SyntheticTrafficEnv()
    with pytest.raises(ValueError):
        env.step(5)
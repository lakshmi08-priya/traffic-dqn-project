import pytest

from evaluation.fixed_time import FixedTimePolicy
from evaluation.metrics import compute_metrics, percent_improvement
from environment.traffic_env import KEEP, SWITCH


def test_compute_metrics_basic():
    m = compute_metrics([10, 20, 30], [50, 60, 70], [2, 4], vehicles_left=5)
    assert m["throughput"] == 3
    assert m["mean_wait"] == pytest.approx(20)
    assert m["mean_travel_time"] == pytest.approx(60)
    assert m["total_wait"] == pytest.approx(60)
    assert m["mean_queue"] == pytest.approx(3)
    assert m["vehicles_left"] == 5


def test_compute_metrics_empty():
    m = compute_metrics([], [], [], 0)
    assert m["throughput"] == 0 and m["mean_wait"] == 0.0 and m["mean_queue"] == 0.0


def test_percent_improvement():
    assert percent_improvement(100, 80) == pytest.approx(20)
    assert percent_improvement(100, 120) == pytest.approx(-20)
    assert percent_improvement(0, 5) == 0.0


def test_fixed_time_policy():
    p = FixedTimePolicy(30)
    assert p.act({"green_elapsed": 25}) == KEEP
    assert p.act({"green_elapsed": 30}) == SWITCH
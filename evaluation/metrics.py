"""Metric calculations for evaluating traffic controllers."""
import xml.etree.ElementTree as ET


def read_tripinfo(path):
    """Return (waiting_times, durations) lists for all finished vehicles."""
    waits, durs = [], []
    for trip in ET.parse(path).getroot().iter("tripinfo"):
        waits.append(float(trip.get("waitingTime")))
        durs.append(float(trip.get("duration")))
    return waits, durs


def compute_metrics(waits, durs, queue_samples, vehicles_left):
    """Combine per-vehicle data and per-step queue samples into summary metrics."""
    n = len(waits)
    return {
        "throughput": n,
        "mean_wait": sum(waits) / n if n else 0.0,
        "mean_travel_time": sum(durs) / n if n else 0.0,
        "total_wait": sum(waits),
        "mean_queue": sum(queue_samples) / len(queue_samples) if queue_samples else 0.0,
        "vehicles_left": vehicles_left,
    }


def percent_improvement(baseline, new):
    """Positive = new is lower (better) than baseline. For metrics where lower is better."""
    if baseline == 0:
        return 0.0
    return (baseline - new) / baseline * 100.0
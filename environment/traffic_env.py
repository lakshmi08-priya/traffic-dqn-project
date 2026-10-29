"""SUMO traffic-signal environment driven through TraCI.

Interface (gym-like):
    state = env.reset(scenario="medium", seed=0)
    state, reward, done, info = env.step(action)   # action: 0 = KEEP, 1 = SWITCH
    state = env.get_state()

Only one TrafficEnv can be active per Python process (it uses TraCI's default connection).
"""
import json
import os

import numpy as np
import sumolib
import traci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "env_config.json")

KEEP, SWITCH = 0, 1

# Signal link indices of our network (from inspect_network.py output).
NS_LINKS = [0, 1, 2, 3, 8, 9, 10, 11]
EW_LINKS = [4, 5, 6, 7, 12, 13, 14, 15]


def load_config(path=None, overrides=None):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = json.load(f)
    if overrides:
        cfg.update(overrides)
    return cfg


class TrafficEnv:
    def __init__(self, config_path=None, overrides=None):
        self.cfg = load_config(config_path, overrides)
        self.lanes = self.cfg["incoming_lanes"]
        self.tls = self.cfg["tls_id"]
        self.greens = self.cfg["green_phases"]      # e.g. [0, 3]
        self.n_phases = 6                            # phases in our signal program
        self.state_dim = len(self.lanes) * 3 + len(self.greens) + 1
        self.n_actions = 2
        self._running = False

    # ------------------------------------------------------------------ reset
    def reset(self, scenario=None, seed=0, tripinfo_path=None):
        """Start a fresh SUMO run and return the initial state."""
        self.close()
        scenario = scenario or self.cfg["default_scenario"]
        cfg_file = os.path.join(ROOT, self.cfg["scenario_dir"], f"{scenario}_seed{seed}.sumocfg")
        if not os.path.exists(cfg_file):
            raise FileNotFoundError(f"Scenario file not found: {cfg_file}")

        sumo = sumolib.checkBinary("sumo")
        cmd = [sumo, "-c", cfg_file, "--no-step-log", "true",
               "--no-warnings", "true", "--seed", str(seed)]
        if tripinfo_path:
            cmd += ["--tripinfo-output", tripinfo_path]
        traci.start(cmd)
        self._running = True

        self.scenario, self.seed = scenario, seed
        self._green_idx = 0            # index into self.greens
        self._green_elapsed = 0        # seconds the current green has lasted
        self._sim_time = traci.simulation.getTime()
        self._done = False
        self._arrived_total = 0
        self.safety_violations = 0

        if traci.trafficlight.getPhase(self.tls) != self.greens[0]:
            traci.trafficlight.setPhase(self.tls, self.greens[0])
        self._hold_green()
        self._prev_wait = self._total_wait()
        return self.get_state()

    # ------------------------------------------------------------------- step
    def step(self, action):
        """Apply KEEP/SWITCH, advance the simulation, return (state, reward, done, info)."""
        if action not in (KEEP, SWITCH):
            raise ValueError(f"Invalid action {action!r}; must be 0 (KEEP) or 1 (SWITCH)")
        if not self._running or self._done:
            raise RuntimeError("Call reset() before step(), and after an episode is done.")

        elapsed = self._green_elapsed
        rejected = action == SWITCH and elapsed < self.cfg["min_green"]
        forced = action == KEEP and elapsed >= self.cfg["max_green"]
        do_switch = (action == SWITCH and not rejected) or forced

        queue_sum, queue_n, arrived = 0.0, 0, 0

        def tick():
            nonlocal queue_sum, queue_n, arrived
            traci.simulationStep()
            self._sim_time = traci.simulation.getTime()
            arrived += traci.simulation.getArrivedNumber()
            queue_sum += self._total_queue()
            queue_n += 1
            self._check_safety()

        if do_switch:
            target = self.greens[1 - self._green_idx]
            traci.trafficlight.setPhase(self.tls, (self.greens[self._green_idx] + 1) % self.n_phases)
            traci.trafficlight.setPhaseDuration(self.tls, self.cfg["yellow_time"])
            guard = 0
            while traci.trafficlight.getPhase(self.tls) != target and self._sim_time < self.cfg["episode_seconds"]:
                tick()
                guard += 1
                if guard > 30:
                    raise RuntimeError("Signal did not reach the next green phase within 30 s")
            self._green_idx = 1 - self._green_idx
            self._green_elapsed = 0

        self._hold_green()
        for _ in range(self.cfg["delta_time"]):
            if self._sim_time >= self.cfg["episode_seconds"]:
                break
            tick()
            self._green_elapsed += 1

        self._done = self._sim_time >= self.cfg["episode_seconds"]
        if not self._done and traci.trafficlight.getPhase(self.tls) != self.greens[self._green_idx]:
            raise RuntimeError("Signal left green unexpectedly; green hold failed")

        cur_wait = self._total_wait()
        reward = (self._prev_wait - cur_wait) / self.cfg["reward_scale"]
        self._prev_wait = cur_wait
        self._arrived_total += arrived

        info = {
            "sim_time": self._sim_time,
            "green_index": self._green_idx,
            "green_elapsed": self._green_elapsed,
            "switched": do_switch,
            "rejected": rejected,
            "forced": forced,
            "total_wait": cur_wait,
            "mean_queue": queue_sum / queue_n if queue_n else 0.0,
            "arrived_step": arrived,
            "arrived_total": self._arrived_total,
            "safety_violations": self.safety_violations,
        }
        return self.get_state(), reward, self._done, info

    # -------------------------------------------------------------- get_state
    def get_state(self):
        cap, wnorm = self.cfg["lane_capacity"], self.cfg["wait_norm"]
        dens, queue, wait = [], [], []
        for lane in self.lanes:
            dens.append(traci.lane.getLastStepVehicleNumber(lane) / cap)
            queue.append(traci.lane.getLastStepHaltingNumber(lane) / cap)
            wait.append(traci.lane.getWaitingTime(lane) / wnorm)
        phase = [1.0 if i == self._green_idx else 0.0 for i in range(len(self.greens))]
        elapsed = [self._green_elapsed / self.cfg["max_green"]]
        state = np.clip(np.array(dens + queue + wait + phase + elapsed, dtype=np.float32), 0.0, 1.0)
        return state

    # ---------------------------------------------------------------- helpers
    def _hold_green(self):
        """Keep SUMO from ending the current green by itself."""
        traci.trafficlight.setPhaseDuration(self.tls, 1000)

    def _total_wait(self):
        return float(sum(traci.lane.getWaitingTime(l) for l in self.lanes))

    def _total_queue(self):
        return float(sum(traci.lane.getLastStepHaltingNumber(l) for l in self.lanes))

    def _check_safety(self):
        """Count a violation if NS and EW links are green at the same moment."""
        s = traci.trafficlight.getRedYellowGreenState(self.tls)
        ns_green = any(s[i] in "Gg" for i in NS_LINKS)
        ew_green = any(s[i] in "Gg" for i in EW_LINKS)
        if ns_green and ew_green:
            self.safety_violations += 1

    def close(self):
        if self._running:
            try:
                traci.close()
            except Exception:
                pass
            self._running = False
    def vehicles_in_network(self):
        return traci.vehicle.getIDCount() if self._running else 0
"""Generate route files (traffic demand) and .sumocfg files for low/medium/high traffic."""
import argparse
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTE_DIR = os.path.join(ROOT, "sumo", "routes")
CFG_DIR = os.path.join(ROOT, "sumo", "scenarios")

# Total vehicles per hour entering the intersection (split equally over 4 arms).
SCENARIOS = {"low": 600, "medium": 1200, "high": 1800}
DURATION = 3600  # seconds of vehicle generation
ARMS = ["N", "E", "S", "W"]

# Destination arm for each movement, taken from the real link list of our network.
TURNS = {
    "N": {"right": "W", "straight": "S", "left": "E"},
    "E": {"right": "N", "straight": "W", "left": "S"},
    "S": {"right": "E", "straight": "N", "left": "W"},
    "W": {"right": "S", "straight": "E", "left": "N"},
}
TURN_PROBS = {"straight": 0.6, "right": 0.2, "left": 0.2}


def generate(scenario, seed):
    """Poisson arrivals on every arm: gaps between cars are exponentially distributed."""
    rng = random.Random(f"{scenario}-{seed}")  # same (scenario, seed) -> same traffic
    rate = SCENARIOS[scenario] / 4 / 3600.0    # vehicles per second per arm
    vehicles = []
    for arm in ARMS:
        t = 0.0
        while True:
            t += rng.expovariate(rate)
            if t >= DURATION:
                break
            turn = rng.choices(list(TURN_PROBS), weights=list(TURN_PROBS.values()))[0]
            vehicles.append((t, arm, TURNS[arm][turn]))
    vehicles.sort()

    name = f"{scenario}_seed{seed}"
    route_path = os.path.join(ROUTE_DIR, f"{name}.rou.xml")
    with open(route_path, "w") as f:
        f.write("<routes>\n")
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" '
                'minGap="2.5" maxSpeed="13.89"/>\n')
        for i, (t, src, dst) in enumerate(vehicles):
            f.write(f'    <vehicle id="v{i}" type="car" depart="{t:.2f}" departLane="best" '
                    f'departSpeed="max"><route edges="{src}_C C_{dst}"/></vehicle>\n')
        f.write("</routes>\n")

    cfg_path = os.path.join(CFG_DIR, f"{name}.sumocfg")
    with open(cfg_path, "w") as f:
        f.write(f"""<configuration>
    <input>
        <net-file value="../network/intersection.net.xml"/>
        <route-files value="../routes/{name}.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{DURATION}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
""")
    return name, len(vehicles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="how many seeds per scenario")
    args = parser.parse_args()
    os.makedirs(ROUTE_DIR, exist_ok=True)
    os.makedirs(CFG_DIR, exist_ok=True)
    for scenario in SCENARIOS:
        for seed in range(args.seeds):
            name, n = generate(scenario, seed)
            print(f"{name}: {n} vehicles")
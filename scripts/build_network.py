"""Build a 4-way signalised intersection from node/edge files using netconvert."""
import os
import subprocess
import sys

import sumolib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_DIR = os.path.join(ROOT, "sumo", "network")
NOD = os.path.join(NET_DIR, "intersection.nod.xml")
EDG = os.path.join(NET_DIR, "intersection.edg.xml")
OUT = os.path.join(NET_DIR, "intersection_raw.net.xml")

ARM_LENGTH = 200   # metres from centre to the end of each road
LANES = 2          # lanes per direction
SPEED = 13.89      # m/s (50 km/h)

ARMS = {"N": (0, ARM_LENGTH), "E": (ARM_LENGTH, 0), "S": (0, -ARM_LENGTH), "W": (-ARM_LENGTH, 0)}


def write_nodes():
    lines = ['<nodes>', '    <node id="C" x="0" y="0" type="traffic_light"/>']
    for name, (x, y) in ARMS.items():
        lines.append(f'    <node id="{name}" x="{x}" y="{y}" type="priority"/>')
    lines.append('</nodes>')
    with open(NOD, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_edges():
    lines = ['<edges>']
    for name in ARMS:
        lines.append(f'    <edge id="{name}_C" from="{name}" to="C" numLanes="{LANES}" speed="{SPEED}"/>')
        lines.append(f'    <edge id="C_{name}" from="C" to="{name}" numLanes="{LANES}" speed="{SPEED}"/>')
    lines.append('</edges>')
    with open(EDG, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(NET_DIR, exist_ok=True)
    write_nodes()
    write_edges()
    netconvert = sumolib.checkBinary("netconvert")
    cmd = [netconvert, "--node-files", NOD, "--edge-files", EDG,
           "--no-turnarounds", "true", "-o", OUT]
    print(">", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-800:])
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(f"netconvert failed with code {result.returncode}")
    print("Raw network written:", OUT)


if __name__ == "__main__":
    main()
"""Write our 6-phase signal program and bake it into the final network."""
import os
import subprocess
import sys

import sumolib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_DIR = os.path.join(ROOT, "sumo", "network")
RAW = os.path.join(NET_DIR, "intersection_raw.net.xml")
TLS = os.path.join(NET_DIR, "tls.xml")
FINAL = os.path.join(NET_DIR, "intersection.net.xml")

NS_GREEN = "GGGgrrrrGGGgrrrr"
NS_YELLOW = "yyyyrrrryyyyrrrr"
EW_GREEN = "rrrrGGGgrrrrGGGg"
EW_YELLOW = "rrrryyyyrrrryyyy"
ALL_RED = "r" * 16

GREEN, YELLOW, ALL_RED_T = 30, 4, 2

PHASES = [
    (GREEN, NS_GREEN), (YELLOW, NS_YELLOW), (ALL_RED_T, ALL_RED),
    (GREEN, EW_GREEN), (YELLOW, EW_YELLOW), (ALL_RED_T, ALL_RED),
]


def main():
    lines = ["<tlLogics>", '    <tlLogic id="C" type="static" programID="0" offset="0">']
    for dur, state in PHASES:
        lines.append(f'        <phase duration="{dur}" state="{state}"/>')
    lines += ["    </tlLogic>", "</tlLogics>"]
    with open(TLS, "w") as f:
        f.write("\n".join(lines) + "\n")

    netconvert = sumolib.checkBinary("netconvert")
    cmd = [netconvert, "-s", RAW, "--tllogic-files", TLS, "-o", FINAL]
    print(">", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-600:])
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(f"netconvert failed with code {result.returncode}")
    print("Final network written:", FINAL)


if __name__ == "__main__":
    main()
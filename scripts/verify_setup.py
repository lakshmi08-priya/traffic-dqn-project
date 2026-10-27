"""Stage 2 check: are all packages installed, and can Python drive SUMO via TraCI?"""
import importlib
import os
import subprocess
import sys
import tempfile


def check_imports():
    print(f"Python: {sys.version.split()[0]}")
    for name in ["numpy", "pandas", "matplotlib", "pytest", "torch", "traci", "sumolib"]:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "n/a")
            print(f"  OK   {name:<11} {version}")
        except Exception as exc:
            print(f"  FAIL {name:<11} {exc}")
            return False
    return True


def check_torch():
    import torch
    x = torch.rand(3, 4)
    y = torch.nn.Linear(4, 2)(x)
    print(f"  torch forward pass OK, output shape {tuple(y.shape)}")


def check_traci():
    import sumolib
    import traci

    print(f"  SUMO_HOME = {os.environ.get('SUMO_HOME')}")
    netgenerate = sumolib.checkBinary("netgenerate")
    sumo = sumolib.checkBinary("sumo")

    with tempfile.TemporaryDirectory() as tmp:
        net = os.path.join(tmp, "test.net.xml")
        subprocess.run(
            [netgenerate, "--grid", "--grid.number", "2", "-o", net],
            check=True, capture_output=True,
        )
        traci.start([sumo, "-n", net, "--end", "10", "--no-step-log", "true"])
        for _ in range(3):
            traci.simulationStep()
        sim_time = traci.simulation.getTime()
        traci.close()
    print(f"  TraCI OK: stepped SUMO, simulation time = {sim_time}")


if __name__ == "__main__":
    ok = check_imports()
    if ok:
        check_torch()
        check_traci()
        print("\nALL CHECKS PASSED")
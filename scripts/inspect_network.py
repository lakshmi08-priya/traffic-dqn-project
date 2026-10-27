"""Print the junction, edges and traffic-light links of a SUMO network."""
import sys

import sumolib

path = sys.argv[1] if len(sys.argv) > 1 else "sumo/network/intersection_raw.net.xml"
net = sumolib.net.readNet(path, withPrograms=True)

print("Nodes:", sorted(n.getID() for n in net.getNodes()))
print("Edges:", sorted(e.getID() for e in net.getEdges()))

for tls in net.getTrafficLights():
    print(f"\nTraffic light: {tls.getID()}")
    for prog_id, prog in tls.getPrograms().items():
        print(f"  Program '{prog_id}' phases:")
        for i, ph in enumerate(prog.getPhases()):
            print(f"    {i}: duration={ph.duration} state={ph.state}")
    print("  Links (index: from-lane -> to-lane):")
    for in_lane, out_lane, link_no in sorted(tls.getConnections(), key=lambda c: c[2]):
        print(f"    {link_no}: {in_lane.getID()} -> {out_lane.getID()}")
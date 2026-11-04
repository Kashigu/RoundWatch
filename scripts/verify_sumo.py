"""Verify Eclipse SUMO network parsing for Town03 roundabout."""

from pathlib import Path
import sumolib


def verify_sumo_network(net_file: str = "data/sumo/Town03.net.xml") -> bool:
    print(f"Reading SUMO network from {net_file}...")
    p = Path(net_file)
    if not p.exists():
        print(f"[ERROR] Net file {net_file} does not exist!")
        return False

    net = sumolib.net.readNet(str(p))
    nodes = net.getNodes()
    edges = net.getEdges()
    roundabouts = net.getRoundabouts()

    print(f"[SUCCESS] SUMO Network Loaded!")
    print(f"Total Network Nodes: {len(nodes)}")
    print(f"Total Road Edges: {len(edges)}")
    print(f"Roundabout structures identified in network: {len(roundabouts)}")

    if roundabouts:
        r = roundabouts[0]
        print(f"Roundabout Edge IDs: {r.getEdges()}")
        print(f"Roundabout Node IDs: {r.getNodes()}")

    return True


if __name__ == "__main__":
    verify_sumo_network()


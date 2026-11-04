import xml.etree.ElementTree as ET
import math

tree = ET.parse("data/sumo/Town03.net.xml")
root = tree.getroot()

# Let's inspect edges 30, 34, -12 and see their lengths, lanes, shapes
for edge_id in ['30', '34', '-12', '16', '46', '15', '-38']:
    edge = root.find(f".//edge[@id='{edge_id}']")
    if edge is not None:
        lanes = edge.findall("lane")
        print(f"Edge {edge_id:6s}: from={edge.attrib.get('from'):6s} to={edge.attrib.get('to'):6s} num_lanes={len(lanes)}")
        for l in lanes:
            length = float(l.attrib.get('length', 0))
            shape = l.attrib.get('shape', '').split()
            print(f"  Lane {l.attrib.get('id'):10s}: length={length:.1f}m, start={shape[0] if shape else None}, end={shape[-1] if shape else None}")

# Find edges that connect to the beginning of edge 30, 34, -12
print("\nEdges feeding INTO 30, 34, -12:")
for conn in root.findall("connection"):
    to_edge = conn.attrib.get("to")
    if to_edge in ['30', '34', '-12']:
        print(f"  {conn.attrib.get('from')} -> {to_edge}")


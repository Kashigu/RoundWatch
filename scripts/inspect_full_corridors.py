import xml.etree.ElementTree as ET

tree = ET.parse("data/sumo/Town03.net.xml")
root = tree.getroot()

candidates = ['-8', '8', '31', '-31', '-24', '24', '15', '46', '-38', '1191', '1682', '38.12']
for cid in candidates:
    edge = root.find(f".//edge[@id='{cid}']")
    if edge is not None:
        lanes = edge.findall("lane")
        length = float(lanes[0].attrib.get('length', 0)) if lanes else 0
        print(f"Edge {cid:8s}: from={edge.attrib.get('from'):6s} to={edge.attrib.get('to'):6s} length={length:6.1f}m lanes={len(lanes)}")

print("\nConnections FROM outgoing edges (15, 46, -38, 16):")
for conn in root.findall("connection"):
    f = conn.attrib.get("from")
    if f in ['15', '46', '-38', '16']:
        print(f"  {f} -> {conn.attrib.get('to')}")


import xml.etree.ElementTree as ET

tree = ET.parse('data/sumo/Town03.net.xml')
root = tree.getroot()

edges_to_check = ['-35', '-36', '-37', '-38', '-12', '-13', '-14', '-25', '28']

print('--- EDGES ---')
for edge in root.findall('edge'):
    eid = edge.get('id')
    if eid in edges_to_check:
        print(f"Edge {eid}: from={edge.get('from')} to={edge.get('to')} func={edge.get('function')}")
        for lane in edge.findall('lane'):
            print(f"   Lane {lane.get('id')}: shape={lane.get('shape')} length={lane.get('length')}")

print('\n--- CONNECTIONS involving roundabout edges ---')
for conn in root.findall('connection'):
    f = conn.get('from')
    t = conn.get('to')
    if f in edges_to_check or t in edges_to_check:
        print(f"Conn from {f} (lane {conn.get('fromLane')}) to {t} (lane {conn.get('toLane')}) via {conn.get('via')}")


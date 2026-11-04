import xml.etree.ElementTree as ET

tree = ET.parse('data/sumo/Town03.net.xml')
root = tree.getroot()

print("=== JUNCTION 1469 ===")
for j in root.findall('junction'):
    if j.get('id') in ['1469', '38.12', '37.36', '1654']:
        print(f"Junction {j.get('id')}: type={j.get('type')} x={j.get('x')} y={j.get('y')} incLanes={j.get('incLanes')} intLanes={j.get('intLanes')}")

print("\n=== INTERNAL LANES for 1469 and 38.12 ===")
for edge in root.findall('edge'):
    if edge.get('function') == 'internal':
        eid = edge.get('id')
        if any(k in eid for k in ['1469', '38.12']):
            for lane in edge.findall('lane'):
                print(f"Internal lane {lane.get('id')}: shape={lane.get('shape')} length={lane.get('length')}")


import xml.etree.ElementTree as ET

tree = ET.parse('data/sumo/Town03.net.xml')
root = tree.getroot()

# New shapes
shape_3_0 = "153.30,221.94 143.18,220.12 135.79,215.12 131.15,206.93 129.24,195.56 129.40,193.63 130.46,190.19 133.22,185.83"
shape_3_1 = "153.37,218.44 144.70,216.85 138.37,212.49 134.38,205.34 132.74,195.42 132.77,194.83 133.63,191.77 136.03,187.91"
shape_1_0 = "145.59,228.46 142.41,220.56 136.97,212.06 131.74,203.53 129.24,195.56 129.40,193.63 130.46,190.19 133.22,185.83"
shape_1_1 = "149.04,227.90 145.88,220.07 140.44,211.70 135.23,203.32 132.74,195.42 132.77,194.83 133.63,191.77 136.03,187.91"

# 1. Update internal edges :1469_1 and :1469_3
for edge in root.findall('edge'):
    eid = edge.get('id')
    if eid == ':1469_1':
        for lane in edge.findall('lane'):
            if lane.get('id') == ':1469_1_0':
                lane.set('shape', shape_1_0)
                lane.set('length', '47.66')
                print("Updated :1469_1_0")
            elif lane.get('id') == ':1469_1_1':
                lane.set('shape', shape_1_1)
                lane.set('length', '44.89')
                print("Updated :1469_1_1")
    elif eid == ':1469_3':
        for lane in edge.findall('lane'):
            if lane.get('id') == ':1469_3_0':
                lane.set('shape', shape_3_0)
                lane.set('length', '50.84')
                print("Updated :1469_3_0")
            elif lane.get('id') == ':1469_3_1':
                lane.set('shape', shape_3_1)
                lane.set('length', '43.06')
                print("Updated :1469_3_1")

# 2. Update connections
for conn in root.findall('connection'):
    f = conn.get('from')
    t = conn.get('to')
    if f in ['-25', '-37'] and t == '-38':
        conn.set('to', '-13')
        print(f"Updated conn from {f} to -13 (lane {conn.get('fromLane')})")
    elif f in [':1469_1', ':1469_3'] and t == '-38':
        conn.set('to', '-13')
        print(f"Updated internal conn from {f} to -13 (lane {conn.get('fromLane')})")

tree.write('data/sumo/Town03.net.xml', encoding='utf-8', xml_declaration=True)
print("Successfully saved data/sumo/Town03.net.xml via ElementTree!")


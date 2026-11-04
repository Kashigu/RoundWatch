import xml.etree.ElementTree as ET

tree = ET.parse('data/sumo/Town03.net.xml')
root = tree.getroot()

print("Junction 1469:")
for j in root.findall('junction'):
    if j.get('id') == '1469':
        print(ET.tostring(j, encoding='utf-8').decode('utf-8'))

print("\nEdges with function=internal for 1469:")
for e in root.findall('edge'):
    if e.get('id').startswith(':1469_'):
        print(ET.tostring(e, encoding='utf-8').decode('utf-8'))

print("\nConnections from 1469:")
for c in root.findall('connection'):
    f = c.get('from')
    t = c.get('to')
    if f in ['-37', '-25', '-12', '61'] or t in ['-38', '-13', '-61'] or ':1469_' in f or ':1469_' in (c.get('via') or ''):
        print(ET.tostring(c, encoding='utf-8').decode('utf-8'))


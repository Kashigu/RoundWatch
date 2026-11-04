import xml.etree.ElementTree as ET

tree = ET.parse('data/sumo/Town03.net.xml')
root = tree.getroot()

print("Connections from -37, -25, -12, 61:")
for c in root.findall('connection'):
    f = c.get('from')
    if f in ['-37', '-25', '-12', '61']:
        print(f"<connection from='{f}' to='{c.get('to')}' fromLane='{c.get('fromLane')}' toLane='{c.get('toLane')}' via='{c.get('via')}' dir='{c.get('dir')}' state='{c.get('state')}'/>")


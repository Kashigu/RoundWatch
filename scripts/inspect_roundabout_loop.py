import xml.etree.ElementTree as ET

tree = ET.parse("data/sumo/Town03.net.xml")
root = tree.getroot()

rb_edges = {'-36', '-37', '-9', '-10', '-11', '-13', '-14', '-35'}

print("Circulatory edge-to-edge connections:")
for conn in root.findall("connection"):
    f = conn.attrib.get("from")
    t = conn.attrib.get("to")
    if f in rb_edges:
        print(f"  {f} -> {t} (fromLane={conn.attrib.get('fromLane')}, toLane={conn.attrib.get('toLane')})")


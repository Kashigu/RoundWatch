import xml.etree.ElementTree as ET
import numpy as np

tree = ET.parse("data/sumo/Town03.net.xml")
root = tree.getroot()

# Find roundabout element
rb = root.find("roundabout")
nodes = rb.attrib["nodes"].split()
edges = rb.attrib["edges"].split()

print("Roundabout edges:", edges)
print("Roundabout nodes:", nodes)

# Get node positions
node_coords = []
for node_id in nodes:
    junction = root.find(f".//junction[@id='{node_id}']")
    if junction is not None:
        x = float(junction.attrib["x"])
        y = float(junction.attrib["y"])
        node_coords.append((node_id, x, y))
        print(f"Node {node_id:6s}: x={x:8.2f}, y={y:8.2f}")

coords_arr = np.array([(x, y) for _, x, y in node_coords])
cx = float(np.mean(coords_arr[:, 0]))
cy = float(np.mean(coords_arr[:, 1]))
radii = np.hypot(coords_arr[:, 0] - cx, coords_arr[:, 1] - cy)

print("\n" + "="*50)
print(f"EXACT ROUNDABOUT CENTER in SUMO coordinates:")
print(f"  Center X = {cx:.2f}, Center Y = {cy:.2f}")
print(f"  Mean Radius = {np.mean(radii):.2f}m (min={np.min(radii):.2f}m, max={np.max(radii):.2f}m)")
print("="*50)

# Check OpenDrive vs SUMO coordinate transform in netconvert
# Let's inspect location element in net.xml
location = root.find("location")
print("Location element:", location.attrib)


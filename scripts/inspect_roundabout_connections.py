import xml.etree.ElementTree as ET

tree = ET.parse("data/sumo/Town03.net.xml")
root = tree.getroot()

rb = root.find("roundabout")
rb_edges = set(rb.attrib["edges"].split())
rb_nodes = set(rb.attrib["nodes"].split())

print("Roundabout edges:", rb_edges)

# Find all edges entering the roundabout nodes
entering_edges = set()
exiting_edges = set()

for edge in root.findall("edge"):
    edge_id = edge.attrib.get("id")
    if edge_id and not edge_id.startswith(":"):
        from_node = edge.attrib.get("from")
        to_node = edge.attrib.get("to")
        if to_node in rb_nodes and edge_id not in rb_edges:
            entering_edges.add(edge_id)
        if from_node in rb_nodes and edge_id not in rb_edges:
            exiting_edges.add(edge_id)

print(f"\nEntering edges ({len(entering_edges)}):", sorted(list(entering_edges)))
print(f"Exiting edges ({len(exiting_edges)}):", sorted(list(exiting_edges)))

# Find connections from entering edges to roundabout edges
print("\nConnections from entering edges to roundabout:")
for conn in root.findall("connection"):
    from_edge = conn.attrib.get("from")
    to_edge = conn.attrib.get("to")
    if from_edge in entering_edges and to_edge in rb_edges:
        print(f"  Entry {from_edge} -> {to_edge} (lane {conn.attrib.get('fromLane')} -> {conn.attrib.get('toLane')})")


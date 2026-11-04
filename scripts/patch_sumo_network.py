import shutil
import re

net_path = "data/sumo/Town03.net.xml"
backup_path = "data/sumo/Town03.net.xml.bak"

shutil.copyfile(net_path, backup_path)
print(f"Backed up {net_path} to {backup_path}")

with open(net_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update :1469_1 lanes (from -25 to -13)
old_1469_1 = """<edge id=":1469_1" function="internal">
        <lane id=":1469_1_0" index="0" disallow="pedestrian tram rail_urban rail rail_electric rail_fast ship container cable_car subway aircraft wheelchair scooter drone" speed="13.89" length="36.77" width="3.50" shape="145.59,228.46 142.41,220.56 136.97,212.06 131.74,203.53 129.24,195.56" />
        <lane id=":1469_1_1" index="1" disallow="pedestrian tram rail_urban rail rail_electric rail_fast ship container cable_car subway aircraft wheelchair scooter drone" speed="13.89" length="36.77" width="3.50" shape="149.04,227.90 145.88,220.07 140.44,211.70 135.23,203.32 132.74,195.42" />
    </edge>"""

new_1469_1 = """<edge id=":1469_1" function="internal">
        <lane id=":1469_1_0" index="0" disallow="pedestrian tram rail_urban rail rail_electric rail_fast ship container cable_car subway aircraft wheelchair scooter drone" speed="13.89" length="47.66" width="3.50" shape="145.59,228.46 142.41,220.56 136.97,212.06 131.74,203.53 129.24,195.56 129.40,193.63 130.46,190.19 133.22,185.83" />
        <lane id=":1469_1_1" index="1" disallow="pedestrian tram rail_urban rail rail_electric rail_fast ship container cable_car subway aircraft wheelchair scooter drone" speed="13.89" length="44.89" width="3.50" shape="149.04,227.90 145.88,220.07 140.44,211.70 135.23,203.32 132.74,195.42 132.77,194.83 133.63,191.77 136.03,187.91" />
    </edge>"""

assert old_1469_1 in content, "old_1469_1 not found in content!"
content = content.replace(old_1469_1, new_1469_1)

# 2. Update :1469_3 lanes (from -37 to -13)
old_1469_3 = """<edge id=":1469_3" function="internal">
        <lane id=":1469_3_0" index="0" disallow="pedestrian tram rail_urban rail rail_electric rail_fast ship container cable_car subway aircraft wheelchair scooter drone" speed="13.48" length="40.15" width="3.50" shape="153.30,221.94 143.18,220.12 135.79,215.12 131.15,206.93 129.24,195.56" />
        <lane id=":1469_3_1" index="1" disallow="pedestrian tram rail_urban rail rail_electric rail_fast ship container cable_car subway aircraft wheelchair scooter drone" speed="12.57" length="34.75" width="3.50" shape="153.37,218.44 144.70,216.85 138.37,212.49 134.38,205.34 132.74,195.42" />
    </edge>"""

new_1469_3 = """<edge id=":1469_3" function="internal">
        <lane id=":1469_3_0" index="0" disallow="pedestrian tram rail_urban rail rail_electric rail_fast ship container cable_car subway aircraft wheelchair scooter drone" speed="13.48" length="50.84" width="3.50" shape="153.30,221.94 143.18,220.12 135.79,215.12 131.15,206.93 129.24,195.56 129.40,193.63 130.46,190.19 133.22,185.83" />
        <lane id=":1469_3_1" index="1" disallow="pedestrian tram rail_urban rail rail_electric rail_fast ship container cable_car subway aircraft wheelchair scooter drone" speed="12.57" length="43.06" width="3.50" shape="153.37,218.44 144.70,216.85 138.37,212.49 134.38,205.34 132.74,195.42 132.77,194.83 133.63,191.77 136.03,187.91" />
    </edge>"""

assert old_1469_3 in content, "old_1469_3 not found in content!"
content = content.replace(old_1469_3, new_1469_3)

# 3. Update connections from -25 and -37 to -13 (instead of -38)
old_conns = """    <connection from="-25" to="-38" fromLane="0" toLane="0" via=":1469_1_0" dir="s" state="m" />
    <connection from="-25" to="-38" fromLane="1" toLane="1" via=":1469_1_1" dir="s" state="m" />
    <connection from="-37" to="-38" fromLane="0" toLane="0" via=":1469_3_0" dir="L" state="M" />
    <connection from="-37" to="-38" fromLane="1" toLane="1" dir="L" state="M" />"""

new_conns = """    <connection from="-25" to="-13" fromLane="0" toLane="0" via=":1469_1_0" dir="s" state="m" />
    <connection from="-25" to="-13" fromLane="1" toLane="1" via=":1469_1_1" dir="s" state="m" />
    <connection from="-37" to="-13" fromLane="0" toLane="0" via=":1469_3_0" dir="L" state="M" />
    <connection from="-37" to="-13" fromLane="1" toLane="1" via=":1469_3_1" dir="L" state="M" />"""

assert old_conns in content, "old_conns not found in content!"
content = content.replace(old_conns, new_conns)

# 4. Update internal connections :1469_1 and :1469_3 to -13
old_int_conns = """    <connection from=":1469_1" to="-38" fromLane="0" toLane="0" dir="s" state="M" />
    <connection from=":1469_1" to="-38" fromLane="1" toLane="1" dir="s" state="M" />
    <connection from=":1469_3" to="-38" fromLane="0" toLane="0" dir="L" state="M" />
    <connection from=":1469_3" to="-38" fromLane="1" toLane="1" dir="L" state="M" />"""

new_int_conns = """    <connection from=":1469_1" to="-13" fromLane="0" toLane="0" dir="s" state="M" />
    <connection from=":1469_1" to="-13" fromLane="1" toLane="1" dir="s" state="M" />
    <connection from=":1469_3" to="-13" fromLane="0" toLane="0" dir="L" state="M" />
    <connection from=":1469_3" to="-13" fromLane="1" toLane="1" dir="L" state="M" />"""

assert old_int_conns in content, "old_int_conns not found in content!"
content = content.replace(old_int_conns, new_int_conns)

with open(net_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Successfully patched {net_path}!")


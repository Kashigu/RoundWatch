import os
import subprocess
import xml.etree.ElementTree as ET

from runtime_paths import configured_path, sumo_binary

tree = ET.parse('data/sumo/roundabout.rou.xml')
root = tree.getroot()
for v in root.findall('vehicle'):
    if v.get('id') == 'veh_08_mustang_cutoff':
        v.set('depart', '22.0')
    if v.get('id') == 'veh_09_mercedes_exit3':
        v.set('depart', '29.0')
    if v.get('id') == 'veh_10_audi_exit1':
        v.set('depart', '34.0')
tree.write('data/sumo/roundabout.rou.xml', encoding='utf-8', xml_declaration=True)

env = os.environ.copy()
env["SUMO_HOME"] = str(configured_path("SUMO_HOME"))

cmd = [str(sumo_binary()), "-c", "data/sumo/roundabout.sumocfg", "--no-step-log"]
res = subprocess.run(cmd, capture_output=True, text=True, env=env)
errs = [l for l in res.stderr.splitlines() if "collision" in l or "Warning" in l and "SUMO_HOME" not in l]
print("Errors/Warnings:", errs)

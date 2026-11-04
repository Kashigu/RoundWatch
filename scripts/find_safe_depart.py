import os
import subprocess
import xml.etree.ElementTree as ET

env = os.environ.copy()
env["SUMO_HOME"] = r"E:\Simulation_Engines\SUMO"

for d8 in [21.0, 22.0, 23.0, 24.0, 28.0, 29.0, 30.0]:
    tree = ET.parse('data/sumo/roundabout.rou.xml')
    root = tree.getroot()
    for v in root.findall('vehicle'):
        if v.get('id') == 'veh_08_mustang_cutoff':
            v.set('depart', str(d8))
        if v.get('id') == 'veh_09_mercedes_exit3':
            v.set('depart', str(max(d8 + 4.0, 31.0)))
        if v.get('id') == 'veh_10_audi_exit1':
            v.set('depart', str(max(d8 + 9.0, 36.0)))
    tree.write('data/sumo/roundabout.rou.xml')
    
    cmd = [r"E:\Simulation_Engines\SUMO\bin\sumo.exe", "-c", "data/sumo/roundabout.sumocfg", "--no-step-log"]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    errs = [l for l in res.stderr.splitlines() if "collision" in l or "Warning" in l and "SUMO_HOME" not in l]
    print(f"d8={d8}: {len(errs)} warnings -> {errs}")


import os
import subprocess

env = os.environ.copy()
env["SUMO_HOME"] = r"E:\Simulation_Engines\SUMO"

cmd = [r"E:\Simulation_Engines\SUMO\bin\sumo.exe", "-c", "data/sumo/roundabout.sumocfg", "--no-step-log"]
res = subprocess.run(cmd, capture_output=True, text=True, env=env)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
print("Return code:", res.returncode)


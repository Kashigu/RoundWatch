import os
import subprocess

from runtime_paths import configured_path, sumo_binary

env = os.environ.copy()
env["SUMO_HOME"] = str(configured_path("SUMO_HOME"))

cmd = [str(sumo_binary()), "-c", "data/sumo/roundabout.sumocfg", "--no-step-log"]
res = subprocess.run(cmd, capture_output=True, text=True, env=env)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
print("Return code:", res.returncode)

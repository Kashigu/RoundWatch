import subprocess
import os

carla_dir = r"E:\Simulation_Engines\CARLA_0.9.15"
carla_exe = os.path.join(carla_dir, "CarlaUE4.exe")

print("Running CarlaUE4.exe with stdout/stderr capture...")
res = subprocess.run([carla_exe, "-carla-server", "-nosound", "-windowed", "-ResX=800", "-ResY=600"],
                     cwd=carla_dir, capture_output=True, text=True, timeout=10)
print("Exit code:", res.returncode)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)


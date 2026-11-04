import subprocess
import os

carla_dir = r"E:\Simulation_Engines\CARLA_0.9.15"
carla_exe = os.path.join(carla_dir, "CarlaUE4.exe")

print("Launching CARLA daemon server on port 2000...")
proc = subprocess.Popen([
    carla_exe,
    "-carla-server",
    "-quality-level=Low",
    "-nosound",
    "-windowed",
    "-ResX=800",
    "-ResY=600"
], cwd=carla_dir)

proc.wait()

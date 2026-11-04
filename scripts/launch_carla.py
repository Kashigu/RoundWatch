import subprocess
import time
import os

carla_dir = r"E:\Simulation_Engines\CARLA_0.9.15"
carla_exe = os.path.join(carla_dir, "CarlaUE4.exe")

print("Launching CARLA...")
proc = subprocess.Popen([
    carla_exe,
    "-quality-level=Low",
    "-carla-server",
    "-nosound",
    "-windowed",
    "-ResX=800",
    "-ResY=600"
], cwd=carla_dir)

print(f"Process PID: {proc.pid}")
for i in range(10):
    time.sleep(1)
    ret = proc.poll()
    if ret is not None:
        print(f"CARLA exited prematurely with code {ret}")
        break
    else:
        print(f"CARLA still running... ({i+1}s)")


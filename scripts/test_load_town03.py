import subprocess
import os
import time
import socket

carla_dir = r"E:\Simulation_Engines\CARLA_0.9.15"
carla_exe = os.path.join(carla_dir, "CarlaUE4.exe")

def is_listening():
    try:
        s = socket.create_connection(("127.0.0.1", 2000), timeout=1.0)
        s.close()
        return True
    except:
        return False

print("Launching CARLA...")
proc = subprocess.Popen([
    carla_exe,
    "-carla-server",
    "-quality-level=Low",
    "-nosound",
    "-windowed",
    "-ResX=800",
    "-ResY=600"
], cwd=carla_dir)

for i in range(25):
    time.sleep(1)
    if is_listening():
        break

import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(60.0)

available_maps = client.get_available_maps()
print("Available maps:", [m for m in available_maps if "Town03" in m])

print("Attempting to load Town03_Opt...")
t0 = time.time()
world = client.load_world("Town03_Opt")
print(f"Loaded Town03_Opt in {time.time()-t0:.1f}s!")
print("Current map:", world.get_map().name)

proc.terminate()
print("Test completed.")


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

print("1. Launching CarlaUE4.exe with cwd=carla_dir...")
proc = subprocess.Popen([
    carla_exe,
    "-carla-server",
    "-quality-level=Low",
    "-nosound",
    "-windowed",
    "-ResX=800",
    "-ResY=600"
], cwd=carla_dir)

print(f"Spawned PID: {proc.pid}")
for i in range(25):
    time.sleep(1)
    if is_listening():
        print(f"Port 2000 connected after {i+1}s!")
        break
else:
    print("Port 2000 did not open.")
    proc.terminate()
    exit(1)

import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(30.0)
world = client.get_world()
print("Connected to CARLA! Map name:", world.get_map().name)

# Now test capturing 4 different camera angles from this world!
bp_lib = world.get_blueprint_library()
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1280")
cam_bp.set_attribute("image_size_y", "720")

# If Town10HD_Opt or whatever map is loaded, let's see what map is loaded
print("Current map is:", world.get_map().name)

proc.terminate()
print("Done test.")


import subprocess
import os
import time
import socket
import shutil
from pathlib import Path
import numpy as np
import cv2

from runtime_paths import artifact_directory, configured_path

carla_root = str(configured_path("CARLA_ROOT"))
carla_exe = os.path.join(carla_root, "CarlaUE4.exe")

def is_open():
    try:
        s = socket.create_connection(("127.0.0.1", 2000), timeout=1.0)
        s.close()
        return True
    except:
        return False

print("1. Launching CARLA server directly...")
proc = subprocess.Popen([
    carla_exe,
    "-carla-server",
    "-quality-level=Low",
    "-nosound",
    "-windowed",
    "-ResX=800",
    "-ResY=600"
], cwd=carla_root)

print(f"CARLA PID: {proc.pid}. Waiting for port 2000...")
connected = False
for i in range(25):
    time.sleep(1.0)
    if is_open():
        print(f"Port 2000 ready after {i+1}s!")
        connected = True
        break

if not connected:
    proc.terminate()
    raise RuntimeError("Failed to connect to CARLA on port 2000.")

import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(60.0)
print("Connecting to world (up to 60s for map load)...")
world = client.get_world()
print("Connected to world:", world.get_map().name)

# Set synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
world.set_weather(carla.WeatherParameters.ClearNoon)

# Clean stale vehicles
for a in world.get_actors().filter("*vehicle*"):
    try:
        a.destroy()
    except:
        pass

# Spawn sample vehicles
bp_lib = world.get_blueprint_library()
spawn_points = [
    (2.0, -15.0, 0.5, 0.0, 90.0, 0.0, "vehicle.audi.tt"),
    (18.0, 6.0, 0.5, 0.0, 180.0, 0.0, "vehicle.mercedes.coupe"),
    (-14.0, 6.0, 0.5, 0.0, 0.0, 0.0, "vehicle.carlamotors.carlacola"),
    (2.0, 27.0, 0.5, 0.0, -90.0, 0.0, "vehicle.mini.cooper_s"),
    (12.0, -8.0, 0.5, 0.0, 135.0, 0.0, "vehicle.ford.mustang"),
]

spawned = []
for sp in spawn_points:
    bp = bp_lib.find(sp[6])
    trans = carla.Transform(
        carla.Location(x=sp[0], y=sp[1], z=sp[2]),
        carla.Rotation(pitch=sp[3], yaw=sp[4], roll=sp[5])
    )
    actor = world.try_spawn_actor(bp, trans)
    if actor:
        spawned.append(actor)

world.tick()

angles = [
    {
        "id": "angle_1_south_view",
        "name": "Angle 1: South Elevated Mast (Baseline)",
        "loc": (2.0, -27.0, 18.0),
        "rot": (-30.0, 90.0, 0.0),
    },
    {
        "id": "angle_2_north_view",
        "name": "Angle 2: North Mast (Opposite Perspective)",
        "loc": (2.0, 39.0, 18.0),
        "rot": (-30.0, -90.0, 0.0),
    },
    {
        "id": "angle_3_east_view",
        "name": "Angle 3: East Avenue Overlook",
        "loc": (35.0, 6.0, 18.0),
        "rot": (-30.0, 180.0, 0.0),
    },
    {
        "id": "angle_4_west_view",
        "name": "Angle 4: West Plaza Perspective",
        "loc": (-31.0, 6.0, 18.0),
        "rot": (-30.0, 0.0, 0.0),
    },
    {
        "id": "angle_5_topdown_aerial",
        "name": "Angle 5: Top-Down Aerial Plan (38m)",
        "loc": (2.0, 6.0, 38.0),
        "rot": (-90.0, 90.0, 0.0),
    },
    {
        "id": "angle_6_isometric_3d",
        "name": "Angle 6: Isometric 3D High Mast (South-East)",
        "loc": (26.0, -18.0, 22.0),
        "rot": (-35.0, 135.0, 0.0),
    },
]

cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1920")
cam_bp.set_attribute("image_size_y", "1080")
cam_bp.set_attribute("fov", "85.0")

output_dir = Path("data/angles")
output_dir.mkdir(parents=True, exist_ok=True)
artifact_dir = artifact_directory()

for item in angles:
    cam_trans = carla.Transform(
        carla.Location(x=item["loc"][0], y=item["loc"][1], z=item["loc"][2]),
        carla.Rotation(pitch=item["rot"][0], yaw=item["rot"][1], roll=item["rot"][2])
    )
    cam = world.spawn_actor(cam_bp, cam_trans)
    holder = {"img": None}

    def on_img(img):
        raw = np.frombuffer(img.raw_data, dtype=np.uint8)
        holder["img"] = raw.reshape((img.height, img.width, 4))[:, :, :3].copy()

    cam.listen(on_img)
    for _ in range(4):
        world.tick()

    if holder["img"] is not None:
        img = holder["img"].copy()
        cv2.rectangle(img, (20, 20), (750, 105), (12, 16, 26), -1)
        cv2.rectangle(img, (20, 20), (750, 105), (56, 189, 248), 2)
        cv2.putText(img, item["name"], (40, 58), cv2.FONT_HERSHEY_DUPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Pos: {item['loc']} | Rot: {item['rot']}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1, cv2.LINE_AA)

        p = output_dir / f"{item['id']}.png"
        cv2.imwrite(str(p), img)
        shutil.copy(str(p), str(artifact_dir / f"{item['id']}.png"))
        print(f"Captured {item['id']}.png")
    cam.destroy()

for a in spawned:
    try:
        a.destroy()
    except:
        pass

# Return to async
settings.synchronous_mode = False
world.apply_settings(settings)

# Clean close CARLA
proc.terminate()
print("All 6 angles captured successfully and CARLA closed!")

"""Captures multiple camera vantage angles of the Town03 roundabout in CARLA.

Angles:
1. South Mast (Current Baseline): (2.0, -27.0, 18.0), pitch -30, yaw 90
2. North Mast (Opposite View)  : (2.0, 39.0, 18.0),  pitch -30, yaw -90
3. East Mast (Avenue Overlook) : (35.0, 6.0, 18.0),  pitch -30, yaw 180
4. West Mast (Plaza View)      : (-31.0, 6.0, 18.0), pitch -30, yaw 0
5. Top-Down Bird's Eye (38m)   : (2.0, 6.0, 38.0),   pitch -90, yaw 90
6. Isometric 3D (South-East)   : (26.0, -18.0, 22.0), pitch -35, yaw 135
"""

import os
import sys
import time
import socket
import subprocess
import shutil
from pathlib import Path
import numpy as np
import cv2

# Ensure CARLA egg / python path
carla_root = r"E:\Simulation_Engines\CARLA_0.9.15"

def is_carla_running(host="127.0.0.1", port=2000):
    try:
        s = socket.create_connection((host, port), timeout=2.0)
        s.close()
        return True
    except Exception:
        return False

def launch_carla():
    if is_carla_running():
        print("[CARLA] Server is already running.")
        return None
    print("[CARLA] Starting CARLA server...")
    carla_exe = os.path.join(carla_root, "CarlaUE4.exe")
    DETACHED = 0x00000008
    NEW_GROUP = 0x00000200
    proc = subprocess.Popen([
        carla_exe,
        "-carla-server",
        "-quality-level=Low",
        "-nosound",
        "-windowed",
        "-ResX=800",
        "-ResY=600"
    ], cwd=carla_root, creationflags=DETACHED | NEW_GROUP, close_fds=True)
    
    print("[CARLA] Waiting for server to initialize on port 2000...")
    for i in range(40):
        time.sleep(1.0)
        if is_carla_running():
            print(f"[CARLA] Server connected successfully after {i+1}s!")
            return proc
    raise TimeoutError("CARLA server failed to start within 40 seconds.")

def main():
    proc = launch_carla()
    import carla

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)

    # Ensure Town03 is loaded
    world = client.get_world()
    if "Town03" not in world.get_map().name:
        print("[CARLA] Loading Town03 map (allowing up to 120s)...")
        world = client.load_world("Town03")
        time.sleep(3.0)

    # Synchronous mode for crisp frames
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # Purge stale actors
    for a in world.get_actors().filter("*vehicle*"):
        try:
            a.destroy()
        except:
            pass

    bp_lib = world.get_blueprint_library()

    # Spawn a few sample vehicles around the roundabout so the view is alive
    spawn_points = [
        # (x, y, z, pitch, yaw, roll, blueprint)
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
            print(f"Spawned {sp[6]} at ({sp[0]}, {sp[1]})")

    world.tick()

    # Camera definitions
    angles = [
        {
            "id": "angle_1_south_view",
            "name": "Angle 1: South Elevated Mast (Baseline)",
            "loc": (2.0, -27.0, 18.0),
            "rot": (-30.0, 90.0, 0.0),
            "desc": "Primary roadside mast facing North. Clean visibility of South entrance and East/West circulation."
        },
        {
            "id": "angle_2_north_view",
            "name": "Angle 2: North Mast (Opposite Perspective)",
            "loc": (2.0, 39.0, 18.0),
            "rot": (-30.0, -90.0, 0.0),
            "desc": "Mast overlooking the North approach heading South. Direct view of oncoming Northern traffic."
        },
        {
            "id": "angle_3_east_view",
            "name": "Angle 3: East Avenue Overlook",
            "loc": (35.0, 6.0, 18.0),
            "rot": (-30.0, 180.0, 0.0),
            "desc": "Looking West across the roundabout from the 4-lane East highway approach."
        },
        {
            "id": "angle_4_west_view",
            "name": "Angle 4: West Plaza Perspective",
            "loc": (-31.0, 6.0, 18.0),
            "rot": (-30.0, 0.0, 0.0),
            "desc": "Looking East from the western approach toward the central monument."
        },
        {
            "id": "angle_5_topdown_aerial",
            "name": "Angle 5: Top-Down Aerial Plan (38m)",
            "loc": (2.0, 6.0, 38.0),
            "rot": (-90.0, 90.0, 0.0),
            "desc": "Orthogonal bird's-eye view showing all 4 arms, lanes, and circulation at 90° overhead."
        },
        {
            "id": "angle_6_isometric_3d",
            "name": "Angle 6: Isometric 3D High Mast (South-East)",
            "loc": (26.0, -18.0, 22.0),
            "rot": (-35.0, 135.0, 0.0),
            "desc": "Corner 3D perspective giving strong spatial depth across both inner and outer circulatory rings."
        },
    ]

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1920")
    cam_bp.set_attribute("image_size_y", "1080")
    cam_bp.set_attribute("fov", "85.0")

    output_dir = Path("data/angles")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(r"C:\Users\orlan\.gemini\antigravity\brain\dc5e9231-1ac9-4f55-b832-205cbb1301e5")

    captured_frames = []

    for item in angles:
        cam_trans = carla.Transform(
            carla.Location(x=item["loc"][0], y=item["loc"][1], z=item["loc"][2]),
            carla.Rotation(pitch=item["rot"][0], yaw=item["rot"][1], roll=item["rot"][2])
        )
        cam = world.spawn_actor(cam_bp, cam_trans)
        holder = {"img": None}

        def on_image(img):
            raw = np.frombuffer(img.raw_data, dtype=np.uint8)
            holder["img"] = raw.reshape((img.height, img.width, 4))[:, :, :3].copy()

        cam.listen(on_image)

        # Tick world twice to let camera render
        for _ in range(4):
            world.tick()

        if holder["img"] is not None:
            # Annotate with title watermark
            img = holder["img"].copy()
            cv2.rectangle(img, (20, 20), (700, 100), (12, 16, 26), -1)
            cv2.rectangle(img, (20, 20), (700, 100), (56, 189, 248), 2)
            cv2.putText(img, item["name"], (40, 55), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"Pos: {item['loc']} | Rot: {item['rot']}", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (148, 163, 184), 1, cv2.LINE_AA)

            file_path = output_dir / f"{item['id']}.png"
            cv2.imwrite(str(file_path), img)
            shutil.copy(str(file_path), str(artifact_dir / f"{item['id']}.png"))
            print(f"[Capture] Saved {item['id']}.png")
            captured_frames.append(item)
        else:
            print(f"[Capture] Failed to grab image for {item['id']}")

        cam.destroy()

    # Clean up spawned vehicles
    for a in spawned:
        try:
            a.destroy()
        except:
            pass

    # Restore asynchronous mode
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("Multi-angle capture complete!")

if __name__ == "__main__":
    main()


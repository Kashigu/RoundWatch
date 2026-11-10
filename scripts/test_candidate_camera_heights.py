import carla
import numpy as np
import cv2
import time
import os
import shutil
from pathlib import Path

from runtime_paths import artifact_directory

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
world = client.get_world()

# Ensure synchronous mode for crisp captures
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

# Spawn sample vehicles all around the roundabout (north, south, east, west, inner, outer)
bp_lib = world.get_blueprint_library()
spawn_points = [
    # South entrance
    (2.0, -18.0, 0.5, 0.0, 90.0, 0.0, "vehicle.audi.tt"),
    # South circulatory lane (directly below where old camera missed)
    (2.0, -9.0, 0.5, 0.0, 90.0, 0.0, "vehicle.ford.mustang"),
    # East circulatory lane
    (18.0, 6.0, 0.5, 0.0, 180.0, 0.0, "vehicle.mercedes.coupe"),
    # West circulatory lane
    (-14.0, 6.0, 0.5, 0.0, 0.0, 0.0, "vehicle.carlamotors.carlacola"),
    # North exit
    (2.0, 27.0, 0.5, 0.0, -90.0, 0.0, "vehicle.mini.cooper_s"),
    # East entry
    (30.0, 8.0, 0.5, 0.0, 180.0, 0.0, "vehicle.volkswagen.t2"),
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

# Candidate camera setups to test
candidates = [
    {
        "id": "cand_0_current_baseline",
        "title": "Current Position (18m, pitch -30 deg)",
        "pos": (2.0, -27.0, 18.0),
        "rot": (-30.0, 90.0, 0.0),
        "fov": 85.0,
        "note": "CURRENT: Bottom south road is cut off, center monument blocks north view."
    },
    {
        "id": "cand_1_steep_south_30m",
        "title": "Option A: Steep South Mast (30m, pitch -50 deg)",
        "pos": (2.0, -36.0, 30.0),
        "rot": (-50.0, 90.0, 0.0),
        "fov": 85.0,
        "note": "Pulled back & elevated: sees right below camera, full south ring visible."
    },
    {
        "id": "cand_2_high_steep_35m",
        "title": "Option B: High Overlook Mast (35m, pitch -60 deg)",
        "pos": (2.0, -22.0, 35.0),
        "rot": (-60.0, 90.0, 0.0),
        "fov": 90.0,
        "note": "Steep 60 deg lookdown: looks over monument, sees all 4 quadrants clearly."
    },
    {
        "id": "cand_3_topdown_aerial_45m",
        "title": "Option C: Bird's Eye Top-Down (45m, pitch -90 deg)",
        "pos": (2.0, 6.0, 45.0),
        "rot": (-90.0, 90.0, 0.0),
        "fov": 85.0,
        "note": "Pure orthogonal 2D overhead: 100% full roundabout, all 4 arms, ZERO occlusion."
    },
    {
        "id": "cand_4_topdown_aerial_35m",
        "title": "Option D: Bird's Eye Focused (35m, pitch -90 deg)",
        "pos": (2.0, 6.0, 35.0),
        "rot": (-90.0, 90.0, 0.0),
        "fov": 90.0,
        "note": "Lower 35m top-down: tight zoom on the circulatory ring and entry lanes."
    },
    {
        "id": "cand_5_isometric_corner_32m",
        "title": "Option E: Isometric Corner Mast (32m, pitch -45 deg)",
        "pos": (32.0, -26.0, 32.0),
        "rot": (-45.0, 140.0, 0.0),
        "fov": 85.0,
        "note": "Diagonal 3D perspective from South-East corner: gives depth across both lanes."
    },
]

cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1920")
cam_bp.set_attribute("image_size_y", "1080")

artifact_dir = artifact_directory()
out_dir = Path("data/camera_tests")
out_dir.mkdir(parents=True, exist_ok=True)

for cand in candidates:
    cam_bp.set_attribute("fov", str(cand["fov"]))
    cam_trans = carla.Transform(
        carla.Location(x=cand["pos"][0], y=cand["pos"][1], z=cand["pos"][2]),
        carla.Rotation(pitch=cand["rot"][0], yaw=cand["rot"][1], roll=cand["rot"][2])
    )
    cam = world.spawn_actor(cam_bp, cam_trans)
    holder = {"img": None}

    def on_img(image):
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        holder["img"] = raw.reshape((image.height, image.width, 4))[:, :, :3].copy()

    cam.listen(on_img)
    for _ in range(5):
        world.tick()

    if holder["img"] is not None:
        img = holder["img"].copy()
        # Watermark
        cv2.rectangle(img, (20, 20), (820, 110), (12, 16, 26), -1)
        cv2.rectangle(img, (20, 20), (820, 110), (56, 189, 248), 2)
        cv2.putText(img, cand["title"], (35, 55), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Pos: {cand['pos']} | Rot: {cand['rot']} | FOV: {cand['fov']}", (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(img, cand["note"], (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (34, 197, 94), 1, cv2.LINE_AA)

        local_file = out_dir / f"{cand['id']}.png"
        cv2.imwrite(str(local_file), img)
        shutil.copy(str(local_file), str(artifact_dir / f"{cand['id']}.png"))
        print(f"Captured: {cand['id']}.png")
    cam.destroy()

# Cleanup vehicles
for a in spawned:
    try:
        a.destroy()
    except:
        pass

settings.synchronous_mode = False
world.apply_settings(settings)
print("All candidate camera views captured successfully!")

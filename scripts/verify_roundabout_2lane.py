"""Verify the 2-lane roundabout in CARLA + SUMO co-simulation (headless, no YOLO).

Loads the OpenDRIVE world into CARLA, runs the SUMO demand through the CARLA bridge, checks that
every CARLA vehicle sits where SUMO puts it, and saves South / North camera frames.

Usage (CARLA server running or auto-started):
    .venv310\\Scripts\\python.exe scripts/verify_roundabout_2lane.py [--seconds 60] [--map roundabout_2lane]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2

from runtime_paths import PROJECT_ROOT, configure_sumo_environment, configured_path

CARLA_ROOT = configured_path("CARLA_ROOT")
SUMO_HOME = configure_sumo_environment()
for module_dir in (str(CARLA_ROOT / "Co-Simulation" / "Sumo"), str(SUMO_HOME / "tools"), str(PROJECT_ROOT)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
os.chdir(PROJECT_ROOT)

import carla  # noqa: E402
from sumo_integration.bridge_helper import BridgeHelper  # noqa: E402
from sumo_integration.carla_simulation import CarlaSimulation  # noqa: E402
from sumo_integration.constants import INVALID_ACTOR_ID  # noqa: E402
from sumo_integration.sumo_simulation import SumoSimulation  # noqa: E402

from src.simulation.camera_rig import CameraRig  # noqa: E402
from src.simulation.maps import dress_world, get_map, load_carla_world  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="roundabout_2lane")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="output/verify_roundabout_2lane")
    args = ap.parse_args()

    spec = get_map(args.map)
    dt = 0.05
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)
    print(f"[verify] Loading CARLA world for '{spec.name}'...")
    load_carla_world(client, spec)
    time.sleep(3.0)

    carla_sim = CarlaSimulation("127.0.0.1", 2000, dt)
    world = carla_sim.world
    scenery = dress_world(client, world, spec)
    if scenery is not None:
        print(f"[scenery] {len(scenery.actors)} props spawned (buildings, markings, islands)")
    world.set_weather(carla.WeatherParameters.ClearNoon)
    print(f"[verify] CARLA map: {world.get_map().name}")

    sumo_sim = SumoSimulation(cfg_file=os.path.abspath(spec.sumo_cfg), step_length=dt,
                              host=None, port=None, sumo_gui=False, client_order=1)
    BridgeHelper.blueprint_library = world.get_blueprint_library()
    BridgeHelper.offset = sumo_sim.get_net_offset()

    cams = {}
    for name, pose in (("south", spec.cam_south), ("north", spec.cam_north)):
        rig = CameraRig(world, image_size=(1280, 720), fov=90.0)
        rig.spawn_camera(pose.position, pose.rotation)
        cams[name] = rig

    sumo2carla = {}
    max_err, spawned, failed = 0.0, 0, 0
    steps = int(args.seconds / dt)
    try:
        for step in range(steps):
            sumo_sim.tick()
            carla_sim.tick()

            for s_id in sumo_sim.spawned_actors - set(sumo2carla):
                sumo_sim.subscribe(s_id)
                s_actor = sumo_sim.get_actor(s_id)
                bp = BridgeHelper.get_carla_blueprint(s_actor, False)
                c_id = INVALID_ACTOR_ID
                if bp is not None:
                    c_id = carla_sim.spawn_actor(bp, BridgeHelper.get_carla_transform(s_actor.transform, s_actor.extent))
                if c_id != INVALID_ACTOR_ID:
                    sumo2carla[s_id] = c_id
                    spawned += 1
                else:
                    failed += 1
                    sumo_sim.unsubscribe(s_id)

            for s_id in sumo_sim.destroyed_actors:
                if s_id in sumo2carla:
                    carla_sim.destroy_actor(sumo2carla.pop(s_id))

            for s_id, c_id in sumo2carla.items():
                s_actor = sumo_sim.get_actor(s_id)
                c_trans = BridgeHelper.get_carla_transform(s_actor.transform, s_actor.extent)
                carla_sim.synchronize_vehicle(c_id, c_trans, None)

            # Position check (one step later CARLA has applied the previous transform)
            if step % 20 == 0:
                for s_id, c_id in sumo2carla.items():
                    actor = world.get_actor(c_id)
                    if actor is None:
                        continue
                    loc = actor.get_location()
                    exp = BridgeHelper.get_carla_transform(sumo_sim.get_actor(s_id).transform,
                                                           sumo_sim.get_actor(s_id).extent).location
                    max_err = max(max_err, math.hypot(loc.x - exp.x, loc.y - exp.y))

            if step % 200 == 100:
                for name, rig in cams.items():
                    if rig.latest_frame is not None:
                        cv2.imwrite(str(out / f"{name}_{step * dt:05.1f}s.png"), rig.latest_frame)
                print(f"[verify] t={step * dt:5.1f}s vehicles in CARLA={len(sumo2carla)}")
    finally:
        for c_id in list(sumo2carla.values()):
            carla_sim.destroy_actor(c_id)
        for rig in cams.values():
            rig.destroy()
        carla_sim.close()
        sumo_sim.close()

    print(f"[verify] spawned in CARLA: {spawned} | failed spawns: {failed} | max CARLA-vs-SUMO position error: {max_err:.2f} m")
    print(f"[verify] frames saved to {out}")
    return 0 if failed == 0 and max_err < 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Record a ground-truth tracking dataset from the CARLA + SUMO co-simulation.

For every simulation frame and each of the two cameras it saves the RGB image (JPEG) and, from a
co-located instance-segmentation camera, the *visible* 2-D box of every vehicle (so occlusion by
the fountain, trees or other vehicles is accounted for). It also stores each vehicle's true world
position and its SUMO id (which carries the compliance label on roundabout_2lane).

Output (default data/eval/<map>/):
    south/000123.jpg, north/000123.jpg   camera frames
    gt.jsonl                             one JSON line per frame (see FRAME FORMAT below)
    meta.json                            map, camera poses, resolution, fps

FRAME FORMAT: {"frame": i, "t": s, "vehicles": [{"sumo_id", "carla_id", "type", "x", "y", "z", "yaw",
               "south": {"vis_box": [x1,y1,x2,y2], "vis_px": n, "full_box": [...]} | null, "north": ...}]}

Usage (CARLA server running):
    .venv310\\Scripts\\python.exe scripts/record_tracking_dataset.py --seconds 90
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np

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

from src.simulation.maps import dress_world, get_map, load_carla_world  # noqa: E402


VEHICLE_TAGS = (14, 15, 16, 18, 19)  # Car, Truck, Bus, Motorcycle, Bicycle


class InstanceMatcher:
    """Learns which instance-segmentation code belongs to which CARLA vehicle actor.

    A code is bound to the actor whose projected 3-D box contains most (>= 60 %) of its pixels;
    once bound it is kept for the rest of the recording.
    """

    def __init__(self) -> None:
        self.code_to_actor: dict = {}

    def visible_boxes(self, inst, tag, full_boxes: dict) -> dict:
        mask = np.isin(tag, VEHICLE_TAGS)
        ys, xs = np.nonzero(mask)
        codes = inst[ys, xs]
        out = {}
        if len(codes) == 0:
            return out
        for code in np.unique(codes):
            sel = codes == code
            cx, cy = xs[sel], ys[sel]
            code = int(code)
            if code not in self.code_to_actor:
                best, best_frac = None, 0.6
                bound = set(self.code_to_actor.values())
                for cid, fb in full_boxes.items():
                    if fb is None or cid in bound:
                        continue
                    inside = np.count_nonzero((cx >= fb[0]) & (cx <= fb[2]) & (cy >= fb[1]) & (cy <= fb[3]))
                    frac = inside / len(cx)
                    if frac > best_frac:
                        best, best_frac = cid, frac
                if best is None:
                    continue
                self.code_to_actor[code] = best
            cid = self.code_to_actor[code]
            out[cid] = ([int(cx.min()), int(cy.min()), int(cx.max()), int(cy.max())], int(len(cx)))
        return out


class SyncCamera:
    """RGB + instance-segmentation camera pair at one pose, read synchronously per world frame."""

    def __init__(self, world, pose, w: int, h: int, fov: float = 90.0) -> None:
        lib = world.get_blueprint_library()
        tf = carla.Transform(carla.Location(*pose.position),
                             carla.Rotation(pitch=pose.rotation[0], yaw=pose.rotation[1], roll=pose.rotation[2]))
        self.w, self.h, self.fov = w, h, fov
        self.sensors, self.queues = [], []
        for kind in ("sensor.camera.rgb", "sensor.camera.instance_segmentation"):
            bp = lib.find(kind)
            bp.set_attribute("image_size_x", str(w))
            bp.set_attribute("image_size_y", str(h))
            bp.set_attribute("fov", str(fov))
            s = world.spawn_actor(bp, tf)
            q: queue.Queue = queue.Queue()
            s.listen(q.put)
            self.sensors.append(s)
            self.queues.append(q)
        f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
        self.K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
        self.world_to_cam = np.array(self.sensors[0].get_transform().get_inverse_matrix())

    def read(self, frame: int, timeout: float = 5.0):
        out = []
        for q in self.queues:
            while True:
                img = q.get(timeout=timeout)
                if img.frame >= frame:
                    break
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            out.append(arr)
        rgb = out[0][:, :, :3].copy()
        seg = out[1]
        # CARLA 0.9.15 instance segmentation (BGRA buffer): R = semantic tag, G + 256 * B = a stable
        # per-object code (NOT the actor id; it is matched to actors by InstanceMatcher)
        inst = seg[:, :, 1].astype(np.int32) + (seg[:, :, 0].astype(np.int32) << 8)
        tag = seg[:, :, 2]
        return rgb, inst, tag

    def full_box(self, actor):
        pts = []
        for v in actor.bounding_box.get_world_vertices(actor.get_transform()):
            pc = self.world_to_cam @ np.array([v.x, v.y, v.z, 1.0])
            ps = np.array([pc[1], -pc[2], pc[0]])
            if ps[2] <= 0.5:
                return None
            pr = self.K @ ps
            pts.append((pr[0] / pr[2], pr[1] / pr[2]))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(self.w - 1, max(xs)), min(self.h - 1, max(ys))
        if x2 <= x1 or y2 <= y1:
            return None
        return [int(x1), int(y1), int(x2), int(y2)]

    def destroy(self) -> None:
        for s in self.sensors:
            s.stop()
            s.destroy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="roundabout_2lane")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-seconds", type=float, default=3.0, help="warm-up before recording starts")
    ap.add_argument("--route-seed", type=int, default=None,
                    help="regenerate the SUMO demand with this seed (different traffic, e.g. for training data)")
    ap.add_argument("--weather", default="ClearNoon", help="carla.WeatherParameters preset, e.g. CloudyNoon, WetNoon, ClearSunset")
    args = ap.parse_args()

    spec = get_map(args.map)
    W, H = (int(v) for v in args.res.lower().split("x"))
    out = Path(args.out or f"data/eval/{spec.name}")
    for cam in ("south", "north"):
        (out / cam).mkdir(parents=True, exist_ok=True)
    dt = 0.05

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(180.0)
    load_carla_world(client, spec)
    time.sleep(3.0)
    carla_sim = CarlaSimulation("127.0.0.1", 2000, dt)
    world = client.get_world()
    world.set_weather(getattr(carla.WeatherParameters, args.weather))
    dress_world(client, world, spec)

    sumo_cfg = spec.sumo_cfg
    if args.route_seed is not None:
        import build_roundabout_2lane as builder

        folder = Path(spec.sumo_cfg).parent
        routes = builder.write_routes(seed=args.route_seed, path=folder / f"_gen_seed{args.route_seed}.rou.xml")
        sumo_cfg = str(builder.write_sumocfg(routes=routes.name, path=folder / f"_gen_seed{args.route_seed}.sumocfg",
                                             seed=args.route_seed))
    sumo_sim = SumoSimulation(cfg_file=os.path.abspath(sumo_cfg), step_length=dt,
                              host=None, port=None, sumo_gui=False, client_order=1)
    BridgeHelper.blueprint_library = world.get_blueprint_library()
    BridgeHelper.offset = sumo_sim.get_net_offset()

    cams = {"south": SyncCamera(world, spec.cam_south, W, H), "north": SyncCamera(world, spec.cam_north, W, H)}
    (out / "meta.json").write_text(json.dumps({
        "map": spec.name, "fps": 1.0 / dt, "width": W, "height": H, "fov": 90.0,
        "sumo_cfg": sumo_cfg, "weather": args.weather,
        "cameras": {"south": {"position": spec.cam_south.position, "rotation": spec.cam_south.rotation},
                    "north": {"position": spec.cam_north.position, "rotation": spec.cam_north.rotation}},
    }, indent=2), encoding="utf-8")

    sumo2carla: dict = {}
    matcher = InstanceMatcher()
    gt_f = open(out / "gt.jsonl", "w", encoding="utf-8")
    steps = int((args.seconds + args.skip_seconds) / dt)
    rec = 0
    t0 = time.time()
    try:
        for step in range(steps):
            sumo_sim.tick()
            carla_sim.tick()   # world.tick() inside; sensors render this frame
            frame = world.get_snapshot().frame

            for s_id in sumo_sim.spawned_actors - set(sumo2carla):
                sumo_sim.subscribe(s_id)
                s_actor = sumo_sim.get_actor(s_id)
                bp = BridgeHelper.get_carla_blueprint(s_actor, False)
                c_id = INVALID_ACTOR_ID
                if bp is not None:
                    c_id = carla_sim.spawn_actor(bp, BridgeHelper.get_carla_transform(s_actor.transform, s_actor.extent))
                if c_id != INVALID_ACTOR_ID:
                    sumo2carla[s_id] = c_id
                else:
                    sumo_sim.unsubscribe(s_id)
            for s_id in sumo_sim.destroyed_actors:
                if s_id in sumo2carla:
                    carla_sim.destroy_actor(sumo2carla.pop(s_id))
            for s_id, c_id in sumo2carla.items():
                s_actor = sumo_sim.get_actor(s_id)
                carla_sim.synchronize_vehicle(c_id, BridgeHelper.get_carla_transform(s_actor.transform, s_actor.extent), None)

            images = {name: cam.read(frame) for name, cam in cams.items()}
            if step * dt < args.skip_seconds:
                continue

            vehicles = []
            actors = {c_id: world.get_actor(c_id) for c_id in sumo2carla.values()}
            full = {name: {cid: cam.full_box(a) for cid, a in actors.items() if a is not None} for name, cam in cams.items()}
            visible = {name: matcher.visible_boxes(images[name][1], images[name][2], full[name]) for name in cams}
            for s_id, c_id in sumo2carla.items():
                actor = world.get_actor(c_id)
                if actor is None:
                    continue
                tf = actor.get_transform()
                rec_v = {"sumo_id": s_id, "carla_id": c_id, "type": actor.type_id,
                         "x": round(tf.location.x, 3), "y": round(tf.location.y, 3), "z": round(tf.location.z, 3),
                         "yaw": round(tf.rotation.yaw, 2)}
                for name in cams:
                    fb = full[name].get(c_id)
                    vb = visible[name].get(c_id)
                    if fb is None and vb is None:
                        rec_v[name] = None
                        continue
                    rec_v[name] = {"vis_box": vb[0] if vb else None, "vis_px": vb[1] if vb else 0, "full_box": fb}
                vehicles.append(rec_v)

            for name in cams:
                cv2.imwrite(str(out / name / f"{rec:06d}.jpg"), images[name][0], [cv2.IMWRITE_JPEG_QUALITY, 93])
            gt_f.write(json.dumps({"frame": rec, "t": round(rec * dt, 3), "vehicles": vehicles}) + "\n")
            rec += 1
            if rec % 200 == 0:
                print(f"[record] {rec} frames ({rec * dt:.0f}s sim) | {len(vehicles)} vehicles | {rec / (time.time() - t0):.1f} fps")
    finally:
        gt_f.close()
        for cam in cams.values():
            cam.destroy()
        for c_id in list(sumo2carla.values()):
            carla_sim.destroy_actor(c_id)
        carla_sim.close()
        sumo_sim.close()
    print(f"[record] done: {rec} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

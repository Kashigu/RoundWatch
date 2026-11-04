"""Watch CARLA + SUMO Co-Simulation Live in a Real-Time Desktop Window.

Runs the synchronized co-simulation indefinitely (or until Q/Esc is pressed),
rendering the overhead camera feed directly into an interactive desktop window
with real-time YOLO vehicle detection and Azuma AR corridors.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"E:\Simulation_Engines\SUMO"

sumo_bridge_dir = r"E:\Simulation_Engines\CARLA_0.9.15\Co-Simulation\Sumo"
if sumo_bridge_dir not in sys.path:
    sys.path.insert(0, sumo_bridge_dir)
sys.path.insert(0, os.path.abspath("."))

import carla
from sumo_integration.bridge_helper import BridgeHelper
from sumo_integration.carla_simulation import CarlaSimulation
from sumo_integration.constants import INVALID_ACTOR_ID
from sumo_integration.sumo_simulation import SumoSimulation

from src.compliance.classifier import TrajectoryClassifier
from src.compliance.geometry import RoundaboutGeometry
from src.perception.detector import VehicleDetector
from src.perception.homography import RoadwayHomography
from src.perception.tracker import BoTSORTTracker
from src.simulation.camera_rig import CameraRig
from src.visualization.ar_overlay import AROverlayEngine


def ensure_carla_running() -> None:
    """Auto-launch CARLA server if not already responding on port 2000."""
    import socket
    import subprocess
    try:
        s = socket.create_connection(("127.0.0.1", 2000), timeout=1.0)
        s.close()
        return
    except Exception:
        pass

    print("[Live Monitor] CARLA server not detected on port 2000. Launching CARLA...")
    carla_dir = r"E:\Simulation_Engines\CARLA_0.9.15"
    carla_exe = os.path.join(carla_dir, "CarlaUE4.exe")
    DETACHED = 0x00000008
    NEW_GROUP = 0x00000200
    subprocess.Popen([
        carla_exe,
        "-carla-server",
        "-quality-level=Low",
        "-nosound",
        "-windowed",
        "-ResX=800",
        "-ResY=600"
    ], cwd=carla_dir, creationflags=DETACHED | NEW_GROUP, close_fds=True)

    for i in range(30):
        time.sleep(1.0)
        try:
            s = socket.create_connection(("127.0.0.1", 2000), timeout=1.0)
            s.close()
            print(f"[Live Monitor] CARLA server ready after {i+1}s!")
            return
        except Exception:
            pass
    print("[Live Monitor] Warning: CARLA server took longer than 30s to respond.")


def run_live_watch(
    sumo_cfg: str = "data/sumo/roundabout.sumocfg",
    fps: int = 20,
) -> None:
    dt = 1.0 / fps
    print("=" * 65)
    print("STARTING LIVE CARLA + SUMO ROUNDABOUT MONITOR")
    print("=" * 65)

    ensure_carla_running()

    # 1. Connect to CARLA
    print("[Live Monitor] Connecting to CARLA Town03...")
    init_client = carla.Client("127.0.0.1", 2000)
    init_client.set_timeout(60.0)
    curr_map = init_client.get_world().get_map().name
    if "Town03" not in curr_map:
        print(f"[Live Monitor] Current map is {curr_map}. Loading Town03_Opt...")
        init_client.load_world("Town03_Opt")
        time.sleep(3.0)

    carla_sim = CarlaSimulation("127.0.0.1", 2000, dt)
    world = carla_sim.world

    # Strictly purge any orphan/stale vehicles from previous runs
    print("[Live Monitor] Purging any stale vehicles from CARLA world...")
    world.tick()
    stale_vehicles = world.get_actors().filter("*vehicle*")
    for a in stale_vehicles:
        try:
            a.destroy()
        except Exception:
            pass
    world.tick()

    world.set_weather(carla.WeatherParameters.ClearNoon)

    # 2. Start SUMO
    print("[Live Monitor] Starting SUMO traffic manager...")
    sumo_sim = SumoSimulation(
        cfg_file=os.path.abspath(sumo_cfg),
        step_length=dt,
        host=None,
        port=None,
        sumo_gui=False,
        client_order=1,
    )

    BridgeHelper.blueprint_library = world.get_blueprint_library()
    BridgeHelper.offset = sumo_sim.get_net_offset()

    # 3. Mount Camera on Physical South Highway Light Pole (BP_HighwayLight_multiple_2)
    cam_pos = (0.71, -26.2, 11.5)
    cam_rot = (-28.0, 88.0, 0.0)

    camera_rig = CameraRig(world, image_size=(1920, 1080), fov=90.0)
    camera_rig.spawn_camera(cam_pos, cam_rot)

    # 4. Perception & AR Engine
    detector = VehicleDetector(model_name="yolov8n.pt", confidence_thresh=0.25, device="cuda:0")
    tracker = BoTSORTTracker(max_age=25, min_hits=2, iou_threshold=0.25)
    geom = RoundaboutGeometry()
    homography = RoadwayHomography()
    if Path("configs/homography_town03.json").exists():
        import json
        with open("configs/homography_town03.json") as f:
            cfg = json.load(f)
        homography.H = np.array(cfg["H_matrix"], dtype=np.float64)
        homography.H_inv = np.array(cfg["H_inv_matrix"], dtype=np.float64)

    classifier = TrajectoryClassifier(model_path="models/compliance_mlp.pt", geometry=geom)
    ar_engine = AROverlayEngine(geometry=geom, homography=homography)

    # Window setup
    window_name = "RoundWatch: Live CARLA + SUMO Real-Time Supervision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    sumo2carla_ids: dict = {}
    accumulated_tracks: dict = {}
    compliance_results: dict = {}
    step = 0
    selected_id: Optional[int] = None
    last_active_tracks: List[Dict[str, Any]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal selected_id
        if event == cv2.EVENT_LBUTTONDOWN:
            vx = int(x * (1920.0 / 1280.0))
            vy = int(y * (1080.0 / 720.0))
            clicked = None
            for trk in last_active_tracks:
                bx1, by1, bx2, by2 = trk["bbox"]
                if (bx1 - 20 <= vx <= bx2 + 20) and (by1 - 20 <= vy <= by2 + 20):
                    clicked = trk["track_id"]
                    break
            if clicked is not None:
                selected_id = None if selected_id == clicked else clicked
            else:
                selected_id = None

    cv2.setMouseCallback(window_name, on_mouse)

    print("\n[Live Monitor] LIVE STREAM ACTIVE! Press 'q' or 'ESC' in window to stop.")

    try:
        while True:
            t_start = time.time()
            timestamp = step * dt

            # Advance simulators
            sumo_sim.tick()
            carla_sim.tick()

            # Sync vehicles
            sumo_spawned = sumo_sim.spawned_actors - set(sumo2carla_ids.keys())
            for s_id in sumo_spawned:
                sumo_sim.subscribe(s_id)
                s_actor = sumo_sim.get_actor(s_id)
                bp = BridgeHelper.get_carla_blueprint(s_actor, False)
                if bp is not None:
                    c_trans = BridgeHelper.get_carla_transform(s_actor.transform, s_actor.extent)
                    c_id = carla_sim.spawn_actor(bp, c_trans)
                    if c_id != INVALID_ACTOR_ID:
                        sumo2carla_ids[s_id] = c_id
                else:
                    sumo_sim.unsubscribe(s_id)

            for s_id in sumo_sim.destroyed_actors:
                if s_id in sumo2carla_ids:
                    c_id = sumo2carla_ids.pop(s_id)
                    carla_sim.destroy_actor(c_id)

            for s_id in list(sumo2carla_ids.keys()):
                c_id = sumo2carla_ids[s_id]
                s_actor = sumo_sim.get_actor(s_id)
                c_trans = BridgeHelper.get_carla_transform(s_actor.transform, s_actor.extent)
                carla_sim.synchronize_vehicle(c_id, c_trans, None)

            # Camera frame
            frame = camera_rig.latest_frame
            if frame is None:
                continue

            # Detection & Tracking
            dets = detector.detect(frame)
            active_tracks = tracker.update(dets, current_frame=frame, timestamp=timestamp)
            last_active_tracks = active_tracks

            for trk in active_tracks:
                tid = trk["track_id"]
                bbox = trk["bbox"]
                fu = (bbox[0] + bbox[2]) / 2.0
                fv = float(bbox[3])
                rx, ry = homography.pixel_to_road(fu, fv)

                if tid not in accumulated_tracks:
                    accumulated_tracks[tid] = []
                accumulated_tracks[tid].append((rx, ry, timestamp))
                trk["history_pixels"] = [(int(fu), int(fv))]

                if len(accumulated_tracks[tid]) >= 5:
                    pts = [(pt[0], pt[1]) for pt in accumulated_tracks[tid]]
                    res = classifier.classify_trajectory(pts)
                    res["raw_trajectory"] = accumulated_tracks[tid]
                    compliance_results[tid] = res

            # Render AR
            annotated = ar_engine.render_overlay(
                frame=frame,
                active_tracks=active_tracks,
                compliance_results=compliance_results,
                selected_track_id=selected_id,
                frame_idx=step,
                timestamp=timestamp,
            )

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            step += 1
            # Maintain 20 FPS real-time clock
            time_taken = time.time() - t_start
            sleep_needed = max(0.001, dt - time_taken)
            time.sleep(sleep_needed)

    finally:
        print("[Live Monitor] Shutting down simulation...")
        cv2.destroyAllWindows()
        camera_rig.destroy()
        carla_sim.close()
        sumo_sim.close()
        print("[Live Monitor] Stopped cleanly.")


if __name__ == "__main__":
    run_live_watch()


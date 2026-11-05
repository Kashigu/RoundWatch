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
    if os.path.exists(r"C:\Simulation_Engines\SUMO"):
        os.environ["SUMO_HOME"] = r"C:\Simulation_Engines\SUMO"
    else:
        os.environ["SUMO_HOME"] = r"E:\Simulation_Engines\SUMO"

for base in [r"C:\Simulation_Engines\CARLA_0.9.15", r"E:\Simulation_Engines\CARLA_0.9.15"]:
    sumo_bridge_dir = os.path.join(base, "Co-Simulation", "Sumo")
    if os.path.exists(sumo_bridge_dir) and sumo_bridge_dir not in sys.path:
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
from src.perception.tracker import DynamicMultiCameraTracker
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
    carla_dir = r"C:\Simulation_Engines\CARLA_0.9.15"
    if not os.path.exists(carla_dir):
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

    # 3. Mount Dual Cameras on Physical Highway Light Poles (South & North)
    # South pole on entrance splitter island (0.71, -26.2, 11.5), North pole on exit splitter island (-2.31, 27.2, 11.5)
    cam_pos_south = (0.71, -26.2, 11.5)
    cam_rot_south = (-28.0, 88.0, 0.0)

    cam_pos_north = (-2.31, 27.2, 11.5)
    cam_rot_north = (-28.0, -80.0, 0.0)

    camera_rig_south = CameraRig(world, image_size=(1280, 720), fov=90.0)
    camera_rig_south.spawn_camera(cam_pos_south, cam_rot_south)

    camera_rig_north = CameraRig(world, image_size=(1280, 720), fov=90.0)
    camera_rig_north.spawn_camera(cam_pos_north, cam_rot_north)

    # 4. Perception & Unified Dual-Camera Tracking Engine
    detector = VehicleDetector(model_name="yolov8n.pt", confidence_thresh=0.25, device="cuda:0")
    tracker = DynamicMultiCameraTracker(
        cam_pos_south, cam_rot_south,
        cam_pos_north, cam_rot_north,
        max_age_frames=40,
        min_hits=2,
    )
    geom = RoundaboutGeometry()

    homography_south = RoadwayHomography()
    homography_north = RoadwayHomography()

    if Path("configs/homography_town03.json").exists():
        import json
        with open("configs/homography_town03.json") as f:
            cfg = json.load(f)
        if "cameras_720" in cfg:
            s_cfg = cfg["cameras_720"]["cam_south"]
            n_cfg = cfg["cameras_720"]["cam_north"]
            homography_south.H = np.array(s_cfg["H_matrix"], dtype=np.float64)
            homography_south.H_inv = np.array(s_cfg["H_inv_matrix"], dtype=np.float64)
            homography_north.H = np.array(n_cfg["H_matrix"], dtype=np.float64)
            homography_north.H_inv = np.array(n_cfg["H_inv_matrix"], dtype=np.float64)
        else:
            homography_south.H = np.array(cfg["H_matrix"], dtype=np.float64)
            homography_south.H_inv = np.array(cfg["H_inv_matrix"], dtype=np.float64)
            homography_north = homography_south

    classifier = TrajectoryClassifier(model_path="models/compliance_mlp.pt", geometry=geom)
    ar_engine_south = AROverlayEngine(geometry=geom, homography=homography_south)
    ar_engine_north = AROverlayEngine(geometry=geom, homography=homography_north)

    # Window setup: Side-by-side dual camera composite (2560x720 displayed scaled to 1600x450 or full window)
    window_name = "RoundWatch: Dual-Camera Live CARLA + SUMO Supervision (South & North Poles)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 450)

    sumo2carla_ids: dict = {}
    accumulated_tracks: dict = {}
    compliance_results: dict = {}
    step = 0
    selected_id: Optional[int] = None
    last_active_tracks: List[Dict[str, Any]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal selected_id
        if event == cv2.EVENT_LBUTTONDOWN:
            vx = int(x * (2560.0 / 1600.0))
            vy = int(y * (720.0 / 450.0))
            clicked = None
            for trk in last_active_tracks:
                tid = trk["id"]
                # Check South bounding box
                if trk.get("active_s"):
                    bx = trk["active_s"]
                    if (bx[0] - 25 <= vx <= bx[2] + 25) and (bx[1] - 25 <= vy <= bx[3] + 25):
                        clicked = tid
                        break
                # Check North bounding box
                if trk.get("active_n"):
                    bx = trk["active_n"]
                    if (bx[0] + 1280 - 25 <= vx <= bx[2] + 1280 + 25) and (bx[1] - 25 <= vy <= bx[3] + 25):
                        clicked = tid
                        break
            if clicked is not None:
                selected_id = None if selected_id == clicked else clicked
            else:
                selected_id = None

    cv2.setMouseCallback(window_name, on_mouse)

    print("\n[Live Monitor] UNIFIED DUAL-CAMERA LIVE STREAM ACTIVE! Press 'q' or 'ESC' in window to stop.")

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

            # Retrieve frames from both cameras
            frame_s = camera_rig_south.latest_frame
            frame_n = camera_rig_north.latest_frame
            if frame_s is None or frame_n is None:
                continue

            pane_w = frame_s.shape[1]
            height = frame_s.shape[0]

            # 1. Independent High-Res YOLO Detection per camera
            dets_s = detector.detect(frame_s)
            dets_n = detector.detect(frame_n)

            # 2. 3D World Road-Plane Fusion & Dynamic Multi-Camera Tracking
            # Car IDs are unified and identical across both cameras
            active_tracks = tracker.update(
                dets_s=dets_s,
                dets_n=dets_n,
                frame_idx=step,
                dt=dt,
                pane_w=pane_w,
                height=height,
            )
            last_active_tracks = active_tracks

            # 3. Update trajectories and compliance
            for trk in active_tracks:
                tid = trk["id"]
                rx, ry = trk["road"][0], trk["road"][1]

                if tid not in accumulated_tracks:
                    accumulated_tracks[tid] = []
                accumulated_tracks[tid].append((rx, ry, timestamp))

                if len(accumulated_tracks[tid]) >= 5:
                    pts = [(pt[0], pt[1]) for pt in accumulated_tracks[tid]]
                    res = classifier.classify_trajectory(pts)
                    res["raw_trajectory"] = accumulated_tracks[tid]
                    compliance_results[tid] = res

            # 4. Prepare active tracks for South & North AR overlay engines
            tracks_s = []
            tracks_n = []
            for trk in active_tracks:
                tid = trk["id"]
                cls_name = trk.get("class_name", "car")
                conf = trk.get("conf", 0.8)

                if trk.get("active_s") is not None:
                    tracks_s.append({
                        "track_id": tid,
                        "id": tid,
                        "bbox": trk["active_s"],
                        "class_name": cls_name,
                        "conf": conf,
                        "history_pixels": [],
                    })
                if trk.get("active_n") is not None:
                    tracks_n.append({
                        "track_id": tid,
                        "id": tid,
                        "bbox": trk["active_n"],
                        "class_name": cls_name,
                        "conf": conf,
                        "history_pixels": [],
                    })

            annotated_s = ar_engine_south.render_overlay(
                frame=frame_s,
                active_tracks=tracks_s,
                compliance_results=compliance_results,
                selected_track_id=selected_id,
                frame_idx=step,
                timestamp=timestamp,
            )
            annotated_n = ar_engine_north.render_overlay(
                frame=frame_n,
                active_tracks=tracks_n,
                compliance_results=compliance_results,
                selected_track_id=selected_id,
                frame_idx=step,
                timestamp=timestamp,
            )

            # Watermark panes with unified tracking information
            cv2.putText(annotated_s, "CAM 1: SOUTH POLE (Splitter Island)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(annotated_n, "CAM 2: NORTH POLE (Splitter Island)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            # Combine both annotated camera views side-by-side
            annotated_composite = np.hstack([annotated_s, annotated_n])

            # Draw divider line between cameras
            cv2.line(annotated_composite, (pane_w, 0), (pane_w, height), (255, 255, 255), 2)

            # Header Status Banner
            cv2.rectangle(annotated_composite, (20, height - 70), (700, height - 20), (20, 20, 20), -1)
            cv2.rectangle(annotated_composite, (20, height - 70), (700, height - 20), (70, 70, 70), 1)
            cv2.putText(
                annotated_composite,
                f"Unified Dual-Camera Tracking | Active: {len(active_tracks)} | Dynamic IDs: {tracker.next_id - 1}",
                (35, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                2,
            )

            cv2.imshow(window_name, annotated_composite)
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
        camera_rig_south.destroy()
        camera_rig_north.destroy()
        carla_sim.close()
        sumo_sim.close()
        print("[Live Monitor] Stopped cleanly.")


if __name__ == "__main__":
    run_live_watch()


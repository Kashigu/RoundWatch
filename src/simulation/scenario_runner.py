"""CARLA Simulation Runner and Ground-Truth Trajectory Recorder.

Synchronously steps the CARLA simulator at 20 FPS (0.05s per tick),
captures elevated camera sensor frames, and exports synchronized
bounding boxes and metric coordinates for MOTA/IDsw benchmarking.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import carla
except ImportError:
    carla = None

from src.simulation.camera_rig import CameraRig


class CarlaScenarioRunner:
    """Manages synchronous multi-vehicle simulation runs and sensor recordings in CARLA."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        fps: int = 20,
        camera_height: float = 8.0,
        output_dir: str = "data/recordings",
    ) -> None:
        self.host = host
        self.port = port
        self.fps = fps
        self.dt = 1.0 / fps
        self.camera_height = camera_height
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.traffic_manager: Optional[Any] = None
        self.camera_rig: Optional[CameraRig] = None
        self.spawned_vehicles: List[Any] = []
        self.original_settings: Optional[Any] = None

    def connect(self) -> None:
        """Establish synchronous connection with CARLA simulator."""
        if carla is None:
            raise RuntimeError("carla package is not installed.")

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(15.0)
        self.world = self.client.get_world()

        if "Town03" not in self.world.get_map().name:
            print("Switching world to Town03...")
            self.world = self.client.load_world("Town03")
            time.sleep(2.0)

        # Set synchronous mode (20 FPS)
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(settings)

        # Configure Traffic Manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(42)

        print("[CARLA Runner] Synchronous mode enabled at 20 FPS.")

    def setup_camera(self) -> None:
        """Mount camera rig overlooking Town03 roundabout at specified elevation."""
        self.camera_rig = CameraRig(self.world, image_size=(1920, 1080), fov=85.0)
        
        # Town03 Roundabout center is around X=-6.4, Y=61.0, Z=0.3
        # Inward perimeter south camera looking north into roundabout
        cam_pos = (-6.4, 61.0 - 28.0, 0.3 + self.camera_height)
        cam_rot = (-25.0, 90.0, 0.0)

        self.camera_rig.spawn_camera(cam_pos, cam_rot)
        print(f"[CARLA Runner] Camera mounted at h={self.camera_height}m, position={cam_pos}")

    def spawn_traffic(self, num_vehicles: int = 15) -> None:
        """Spawn vehicles near roundabout and assign autonomous navigation."""
        bp_library = self.world.get_blueprint_library()
        car_bps = bp_library.filter("vehicle.*")

        spawn_points = self.world.get_map().get_spawn_points()
        # Filter spawn points within 55m of roundabout for direct camera view
        nearby_sps = [
            sp for sp in spawn_points
            if 3.0 < math.hypot(sp.location.x - (-6.4), sp.location.y - 61.0) < 58.0
        ]
        np.random.shuffle(nearby_sps)

        for i in range(min(num_vehicles, len(nearby_sps))):
            sp = nearby_sps[i]
            bp = np.random.choice(car_bps)
            if bp.has_attribute("color"):
                color = np.random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)

            vehicle = self.world.try_spawn_actor(bp, sp)
            if vehicle is not None:
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                # Portuguese Highway Code parameters
                self.traffic_manager.distance_to_leading_vehicle(vehicle, 3.5)
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, np.random.uniform(-10.0, 10.0))
                self.spawned_vehicles.append(vehicle)

        print(f"[CARLA Runner] Successfully spawned {len(self.spawned_vehicles)} autonomous vehicles.")

    def run_recording(self, duration_sec: float = 20.0) -> Tuple[Path, Path]:
        """Record video feed and ground-truth metadata synchronously."""
        total_steps = int(duration_sec * self.fps)
        video_path = self.output_dir / f"carla_feed_h{int(self.camera_height)}m.mp4"
        json_path = self.output_dir / f"carla_gt_h{int(self.camera_height)}m.json"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (1920, 1080))

        gt_records: List[Dict[str, Any]] = []

        print(f"[CARLA Runner] Recording {total_steps} frames ({duration_sec}s)...")
        for step in range(total_steps):
            frame_id = self.world.tick()
            timestamp = step * self.dt

            frame = self.camera_rig.get_frame(frame_id=frame_id)
            if frame is not None:
                writer.write(frame)

            # Record ground truth for all visible vehicles
            for v in self.spawned_vehicles:
                if not v.is_alive:
                    continue
                bbox = self.camera_rig.compute_2d_bounding_box(v)
                if bbox is not None:
                    loc = v.get_location()
                    vel = v.get_velocity()
                    speed_kmh = 3.6 * math.hypot(vel.x, vel.y, vel.z)

                    gt_records.append({
                        "frame_idx": step,
                        "timestamp": round(timestamp, 3),
                        "vehicle_id": v.id,
                        "type": v.type_id,
                        "x": round(loc.x, 3),
                        "y": round(loc.y, 3),
                        "z": round(loc.z, 3),
                        "speed_kmh": round(speed_kmh, 1),
                        "bbox": bbox,
                    })

        writer.release()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(gt_records, f, indent=2)

        print(f"[CARLA Runner] Finished! Video saved to {video_path}")
        print(f"[CARLA Runner] Recorded {len(gt_records)} vehicle observation bounding boxes in {json_path}")
        return video_path, json_path

    def cleanup(self) -> None:
        """Destroy spawned vehicles and restore simulator settings."""
        print("[CARLA Runner] Cleaning up actors...")
        if self.camera_rig is not None:
            try:
                self.camera_rig.destroy()
            except Exception as e:
                print(f"[CARLA Runner] Camera cleanup warning: {e}")

        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass

        if self.client is not None and self.spawned_vehicles:
            try:
                batch = [carla.command.DestroyActor(v) for v in self.spawned_vehicles if v.is_alive]
                self.client.apply_batch(batch)
            except Exception as e:
                print(f"[CARLA Runner] Batch destroy warning: {e}")
        self.spawned_vehicles.clear()

        if self.world is not None and self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception as e:
                print(f"[CARLA Runner] Restore settings warning: {e}")
        print("[CARLA Runner] Cleaned up and restored original settings.")


def run_carla_session(duration: float = 15.0, height: float = 8.0) -> None:
    runner = CarlaScenarioRunner(camera_height=height)
    try:
        runner.connect()
        runner.setup_camera()
        runner.spawn_traffic(num_vehicles=18)
        runner.run_recording(duration_sec=duration)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    run_carla_session(duration=15.0, height=8.0)

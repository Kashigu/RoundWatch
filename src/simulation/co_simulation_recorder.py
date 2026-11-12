"""CARLA + SUMO Synchronous Co-Simulation and Ground-Truth Trajectory Recorder.

Synchronizes SUMO traffic demand (Portuguese Highway Code compliant routes and infractions)
with CARLA 0.9.15 3D world physics on Town03 roundabout, capturing synchronized 1080p
elevated camera frames and exact bounding-box ground-truth trajectories at 20 FPS.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from scripts.runtime_paths import configure_sumo_environment, configured_path

# Ensure SUMO and CARLA integration modules are accessible
sumo_home = configure_sumo_environment()
carla_root = configured_path("CARLA_ROOT")
sumo_bridge_dir = carla_root / "Co-Simulation" / "Sumo"
for module_dir in (str(sumo_bridge_dir), str(sumo_home / "tools")):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
sys.path.insert(0, os.path.abspath("."))

import carla
import traci
from sumo_integration.bridge_helper import BridgeHelper
from sumo_integration.carla_simulation import CarlaSimulation
from sumo_integration.constants import INVALID_ACTOR_ID
from sumo_integration.sumo_simulation import SumoSimulation

from src.simulation.camera_rig import CameraRig
from src.simulation.instance_labels import match_vehicle_instances, vehicle_instance_map


class CarlaSumoCoSimulationRecorder:
    """Manages synchronous multi-vehicle CARLA + SUMO co-simulation and recording."""

    def __init__(
        self,
        sumo_cfg: str = "data/sumo/roundabout.sumocfg",
        carla_host: str = "127.0.0.1",
        carla_port: int = 2000,
        fps: int = 20,
        camera_height: float = 12.0,
        output_dir: str = "data/recordings",
        instance_labels: bool = True,
        save_instance_masks: bool = False,
        min_visible_pixels: int = 25,
    ) -> None:
        """
        Args:
            instance_labels: Derive each label box from an instance-segmentation
                camera mounted alongside each RGB camera (visible, occlusion-aware
                boxes). When False, the projected 3D-box hull is used as before.
            save_instance_masks: Also write the raw instance-segmentation frames
                as lossless PNGs, so labels can be re-derived later.
            min_visible_pixels: Vehicles with fewer visible pixels get no label.
        """
        self.sumo_cfg = os.path.abspath(sumo_cfg)
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.fps = fps
        self.dt = 1.0 / fps
        self.camera_height = camera_height
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.instance_labels = instance_labels
        self.save_instance_masks = save_instance_masks and instance_labels
        self.min_visible_pixels = min_visible_pixels
        self.frame_timeout = 10.0  # seconds to wait for a tick's camera images

        self.carla_sim: Optional[CarlaSimulation] = None
        self.sumo_sim: Optional[SumoSimulation] = None
        self.world: Optional[Any] = None
        self.camera_rig_south: Optional[CameraRig] = None
        self.camera_rig_north: Optional[CameraRig] = None
        self.camera_rig: Optional[CameraRig] = None
        self.sumo2carla_ids: Dict[str, int] = {}

    def setup(self) -> None:
        """Start SUMO server and establish synchronous link with CARLA world."""
        print("[Co-Simulation] Connecting to CARLA on Town03...")
        init_client = carla.Client(self.carla_host, self.carla_port)
        init_client.set_timeout(60.0)
        curr_map = init_client.get_world().get_map().name
        if "Town03" not in curr_map:
            print(f"[Co-Simulation] Current map is {curr_map}. Loading Town03_Opt...")
            init_client.load_world("Town03_Opt")
            time.sleep(3.0)

        self.carla_sim = CarlaSimulation(self.carla_host, self.carla_port, self.dt)
        self.world = self.carla_sim.world

        # Strictly purge any orphan/stale vehicles and sensors from previous runs
        print("[Co-Simulation] Purging any stale vehicles from CARLA world...")
        self.world.tick()
        stale_vehicles = self.world.get_actors().filter("*vehicle*")
        for a in stale_vehicles:
            try:
                a.destroy()
            except Exception:
                pass
        self.world.tick()

        # Set clear noon weather for optimal vision illumination
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        print("[Co-Simulation] Starting SUMO co-simulation instance...")
        self.sumo_sim = SumoSimulation(
            cfg_file=self.sumo_cfg,
            step_length=self.dt,
            host=None,
            port=None,
            sumo_gui=False,
            client_order=1,
        )

        # Initialize BridgeHelper blueprints and coordinate offset
        BridgeHelper.blueprint_library = self.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo_sim.get_net_offset()

        # Mount dual cameras on the exact physical high-mast light poles (BP_HighwayLight_multiple)
        # South pole on entrance splitter island (0.71, -28.33), North pole on exit splitter island (-2.31, 29.31)
        cam_pos_south = (0.71, -26.2, 11.5)
        cam_rot_south = (-28.0, 88.0, 0.0)

        cam_pos_north = (-2.31, 27.2, 11.5)
        cam_rot_north = (-28.0, -80.0, 0.0)

        self.camera_rig_south = CameraRig(self.world, image_size=(1920, 1080), fov=90.0)
        # sensor_tick=0: render on every synchronous world tick. Matching the
        # world's 0.05 s delta instead makes sensors skip ticks at random.
        self.camera_rig_south.spawn_camera(
            cam_pos_south, cam_rot_south,
            with_instance_segmentation=self.instance_labels, sensor_tick=0.0,
        )

        self.camera_rig_north = CameraRig(self.world, image_size=(1920, 1080), fov=90.0)
        self.camera_rig_north.spawn_camera(
            cam_pos_north, cam_rot_north,
            with_instance_segmentation=self.instance_labels, sensor_tick=0.0,
        )

        self.camera_rig = self.camera_rig_south
        print(f"[Co-Simulation] Mounted Dual Lamp Pole Cameras at {self.camera_height}m (South & North)")

    def run_recording(self, duration_sec: float = 45.0) -> Tuple[Path, Path]:
        """Synchronously step SUMO + CARLA, record dual-camera video and ground-truth metadata."""
        total_steps = int(duration_sec * self.fps)
        video_path = self.output_dir / "carla_sumo_roundabout.mp4"
        video_south_path = self.output_dir / "carla_sumo_cam_south.mp4"
        video_north_path = self.output_dir / "carla_sumo_cam_north.mp4"
        json_path = self.output_dir / "carla_sumo_gt.json"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Composite side-by-side (2560x720) for interactive desktop monitoring
        writer_comp = cv2.VideoWriter(str(video_path), fourcc, self.fps, (2560, 720))
        writer_south = cv2.VideoWriter(str(video_south_path), fourcc, self.fps, (1920, 1080))
        writer_north = cv2.VideoWriter(str(video_north_path), fourcc, self.fps, (1920, 1080))

        gt_records: List[Dict[str, Any]] = []

        print(f"[Co-Simulation] Recording {total_steps} frames ({duration_sec}s) at {self.fps} FPS...")
        t0 = time.time()

        stats = {"rgb_fallback": 0, "seg_missing": 0, "labels": 0, "hidden": 0, "id_agree": 0}
        mask_dirs: Dict[str, Path] = {}
        if self.save_instance_masks:
            for cam in ("south", "north"):
                mask_dirs[cam] = self.output_dir / f"instance_{cam}"
                mask_dirs[cam].mkdir(parents=True, exist_ok=True)

        for step in range(total_steps):
            timestamp = step * self.dt

            # 1. Advance SUMO simulation
            self.sumo_sim.tick()

            # 2. Advance CARLA simulation synchronously
            self.carla_sim.tick()
            sim_frame = self.world.get_snapshot().frame

            # 3. Fetch the camera images rendered at exactly this tick
            frame_s, seg_s = self._fetch_frames(self.camera_rig_south, sim_frame, stats)
            frame_n, seg_n = self._fetch_frames(self.camera_rig_north, sim_frame, stats)

            if frame_s is not None and frame_n is not None:
                writer_south.write(frame_s)
                writer_north.write(frame_n)

                s_pane = cv2.resize(frame_s, (1280, 720))
                n_pane = cv2.resize(frame_n, (1280, 720))
                comp_frame = np.hstack([s_pane, n_pane])
                writer_comp.write(comp_frame)
            elif frame_s is not None:
                writer_south.write(frame_s)
            elif frame_n is not None:
                writer_north.write(frame_n)

            for cam, seg in (("south", seg_s), ("north", seg_n)):
                if cam in mask_dirs and seg is not None:
                    cv2.imwrite(str(mask_dirs[cam] / f"frame_{step:06d}.png"), seg[:, :, :3])

            # 4. Ground truth for the same tick as the images above
            gt_records.extend(
                self._ground_truth_for_tick(step, timestamp, {"south": seg_s, "north": seg_n}, stats)
            )

            # 5. Synchronize SUMO vehicles to CARLA actors (takes effect next tick)
            sumo_spawned = self.sumo_sim.spawned_actors - set(self.sumo2carla_ids.keys())
            for s_id in sumo_spawned:
                self.sumo_sim.subscribe(s_id)
                s_actor = self.sumo_sim.get_actor(s_id)
                bp = BridgeHelper.get_carla_blueprint(s_actor, False)
                if bp is not None:
                    c_trans = BridgeHelper.get_carla_transform(s_actor.transform, s_actor.extent)
                    c_id = self.carla_sim.spawn_actor(bp, c_trans)
                    if c_id != INVALID_ACTOR_ID:
                        self.sumo2carla_ids[s_id] = c_id
                else:
                    self.sumo_sim.unsubscribe(s_id)

            # Destroy departed vehicles
            for s_id in self.sumo_sim.destroyed_actors:
                if s_id in self.sumo2carla_ids:
                    c_id = self.sumo2carla_ids.pop(s_id)
                    self.carla_sim.destroy_actor(c_id)

            # Update transforms of moving vehicles
            for s_id in list(self.sumo2carla_ids.keys()):
                c_id = self.sumo2carla_ids[s_id]
                s_actor = self.sumo_sim.get_actor(s_id)
                c_trans = BridgeHelper.get_carla_transform(s_actor.transform, s_actor.extent)
                self.carla_sim.synchronize_vehicle(c_id, c_trans, None)

            if step % 50 == 0:
                elapsed = time.time() - t0
                cur_rate = step / elapsed if elapsed > 0 else 0
                print(
                    f"  Tick {step:3d}/{total_steps} | Active Co-Sim Cars: {len(self.sumo2carla_ids):2d}"
                    f" | Speed: {cur_rate:.1f} FPS | Missing frames: rgb {stats['rgb_fallback']}"
                    f", seg {stats['seg_missing']}"
                )

        writer_comp.release()
        writer_south.release()
        writer_north.release()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(gt_records, f, indent=2)

        tot_time = time.time() - t0
        print("\n" + "=" * 60)
        print("CARLA + SUMO CO-SIMULATION RECORDING COMPLETE")
        print("=" * 60)
        print(f"Recorded Dual Video:   {video_path} ({total_steps} frames in {tot_time:.1f}s)")
        print(f"Recorded South Camera: {video_south_path}")
        print(f"Recorded North Camera: {video_north_path}")
        print(f"Ground-Truth Log:      {json_path} ({len(gt_records)} bounding box entries)")
        if stats["rgb_fallback"]:
            print(f"WARNING: {stats['rgb_fallback']} camera frames were not delivered for their tick; "
                  "the latest available frame was written instead.")
        if self.instance_labels:
            print(f"Instance labels:       {stats['labels']} visible boxes, "
                  f"{stats['hidden']} projected-but-hidden vehicles skipped")
            if stats["seg_missing"]:
                print(f"WARNING: {stats['seg_missing']} segmentation frames missing; "
                      "projected boxes were used for those ticks.")
            if stats["labels"]:
                agree = 100.0 * stats["id_agree"] / stats["labels"]
                print(f"Instance ID == actor ID for {agree:.0f}% of matched labels "
                      "(diagnostic only; matching uses box overlap)")
        print("=" * 60)
        return video_path, json_path

    def _fetch_frames(
        self, rig: Optional[CameraRig], sim_frame: int, stats: Dict[str, int]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """RGB and instance-segmentation images rendered at ``sim_frame``."""
        if rig is None:
            return None, None
        # Generous timeout: the first spawn of a new vehicle model can stall
        # rendering for a few seconds while CARLA loads its assets.
        frame = rig.wait_for_frame(sim_frame, timeout=self.frame_timeout)
        if frame is None:
            stats["rgb_fallback"] += 1
            frame = rig.latest_frame
        seg = None
        if self.instance_labels:
            seg = rig.wait_for_instance_frame(sim_frame, timeout=self.frame_timeout)
            if seg is None:
                stats["seg_missing"] += 1
        return frame, seg

    def _ground_truth_for_tick(
        self,
        step: int,
        timestamp: float,
        segs: Dict[str, Optional[np.ndarray]],
        stats: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Label boxes for every live vehicle in both cameras for one tick."""
        actors = {}
        for s_id, cid in self.sumo2carla_ids.items():
            v_actor = self.carla_sim.get_actor(cid)
            if v_actor is not None and v_actor.is_alive:
                actors[cid] = (s_id, v_actor)

        rigs = {"south": self.camera_rig_south, "north": self.camera_rig_north}
        projected: Dict[str, Dict[int, Any]] = {}
        labels: Dict[str, Dict[int, Any]] = {}
        visible: Dict[str, Dict[int, int]] = {}
        source: Dict[str, str] = {}
        for cam, rig in rigs.items():
            projected[cam] = {
                cid: (rig.compute_2d_bounding_box(actor, min_size=0) if rig else None)
                for cid, (_, actor) in actors.items()
            }
            seg = segs.get(cam)
            if self.instance_labels and seg is not None:
                matched = match_vehicle_instances(
                    vehicle_instance_map(seg),
                    projected[cam],
                    min_visible_pixels=self.min_visible_pixels,
                )
                labels[cam] = {cid: list(m.bbox) for cid, m in matched.items()}
                visible[cam] = {cid: m.visible_pixels for cid, m in matched.items()}
                source[cam] = "instance_segmentation"
                stats["labels"] += len(matched)
                stats["hidden"] += sum(
                    1 for cid, box in projected[cam].items() if box is not None and cid not in matched
                )
                stats["id_agree"] += sum(
                    1 for cid, m in matched.items() if m.instance_id == (cid & 0xFFFF)
                )
            else:
                # Legacy behaviour: projected 3D-box hull, at least 10 px each side.
                labels[cam] = {
                    cid: list(box)
                    for cid, box in projected[cam].items()
                    if box is not None and box[2] - box[0] >= 10 and box[3] - box[1] >= 10
                }
                visible[cam] = {}
                source[cam] = "projected"

        records = []
        for cid, (s_id, v_actor) in actors.items():
            bbox_s = labels["south"].get(cid)
            bbox_n = labels["north"].get(cid)
            if bbox_s is None and bbox_n is None:
                continue
            loc = v_actor.get_location()
            vel = v_actor.get_velocity()
            speed_kmh = 3.6 * math.hypot(vel.x, vel.y, vel.z)
            proj_s = projected["south"].get(cid)
            proj_n = projected["north"].get(cid)
            records.append({
                "frame_idx": step,
                "timestamp": round(timestamp, 3),
                "vehicle_id": cid,
                "sumo_id": s_id,
                "type": v_actor.type_id,
                "x": round(loc.x, 3),
                "y": round(loc.y, 3),
                "z": round(loc.z, 3),
                "speed_kmh": round(speed_kmh, 1),
                "bbox_south": bbox_s,
                "bbox_north": bbox_n,
                "bbox": bbox_s or bbox_n,
                "bbox_south_projected": list(proj_s) if proj_s else None,
                "bbox_north_projected": list(proj_n) if proj_n else None,
                "visible_px_south": visible["south"].get(cid),
                "visible_px_north": visible["north"].get(cid),
                "label_source_south": source["south"],
                "label_source_north": source["north"],
            })
        return records

    def cleanup(self) -> None:
        """Clean up simulator actors and close TraCI / CARLA sessions."""
        print("[Co-Simulation] Cleaning up actors...")
        if self.camera_rig_south is not None:
            try:
                self.camera_rig_south.destroy()
            except Exception:
                pass
        if self.camera_rig_north is not None:
            try:
                self.camera_rig_north.destroy()
            except Exception:
                pass

        # The SUMO bridge's close() leaves its vehicles in the CARLA world.
        if self.carla_sim is not None:
            for c_id in list(self.sumo2carla_ids.values()):
                try:
                    self.carla_sim.destroy_actor(c_id)
                except Exception:
                    pass
            self.sumo2carla_ids.clear()

        if self.carla_sim is not None:
            try:
                self.carla_sim.close()
            except Exception:
                pass

        if self.sumo_sim is not None:
            try:
                self.sumo_sim.close()
            except Exception:
                pass
        print("[Co-Simulation] Cleanup complete.")


def run_co_simulation(duration: float = 45.0) -> None:
    recorder = CarlaSumoCoSimulationRecorder(
        sumo_cfg="data/sumo/roundabout.sumocfg",
        fps=20,
        camera_height=12.0,
    )
    try:
        recorder.setup()
        recorder.run_recording(duration_sec=duration)
    finally:
        recorder.cleanup()


if __name__ == "__main__":
    run_co_simulation(duration=45.0)

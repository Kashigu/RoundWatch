"""Virtual Camera Rig for Roundabout Observation in CARLA 0.9.15.

Handles multi-elevation mounting (6m, 8m, 10m), perspective projections,
and automated ground-truth 2D/3D bounding box generation.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import carla
except ImportError:
    carla = None


class CameraRig:
    """Manages virtual overhead/roadside camera sensors attached to CARLA world."""

    def __init__(
        self,
        world: Any,
        image_size: Tuple[int, int] = (1920, 1080),
        fov: float = 90.0,
    ) -> None:
        self.world = world
        self.width, self.height = image_size
        self.fov = fov
        self.sensor_actor: Optional[Any] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.image_queue: queue.Queue = queue.Queue(maxsize=100)
        self.frame_count = 0

        # Optional instance-segmentation sensor at the same pose (exact labels).
        self.instance_sensor: Optional[Any] = None
        # Recent images keyed by simulation frame, for tick-exact retrieval.
        self._frame_cond = threading.Condition()
        self._rgb_by_frame: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._instance_by_frame: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._buffer_size = 64

        # Build camera intrinsic calibration matrix K
        focal_length = self.width / (2.0 * math.tan(fov * math.pi / 360.0))
        self.K = np.array([
            [focal_length, 0.0, self.width / 2.0],
            [0.0, focal_length, self.height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _store(self, buffer: "OrderedDict[int, np.ndarray]", frame_id: int, data: np.ndarray) -> None:
        with self._frame_cond:
            buffer[frame_id] = data
            while len(buffer) > self._buffer_size:
                buffer.popitem(last=False)
            self._frame_cond.notify_all()

    def _wait(
        self, buffer: "OrderedDict[int, np.ndarray]", frame_id: int, timeout: float
    ) -> Optional[np.ndarray]:
        deadline = time.time() + timeout
        with self._frame_cond:
            while frame_id not in buffer:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._frame_cond.wait(remaining)
            return buffer[frame_id]

    def wait_for_frame(self, frame_id: int, timeout: float = 2.0) -> Optional[np.ndarray]:
        """BGR image rendered at simulation frame ``frame_id`` (None on timeout)."""
        return self._wait(self._rgb_by_frame, frame_id, timeout)

    def wait_for_instance_frame(self, frame_id: int, timeout: float = 2.0) -> Optional[np.ndarray]:
        """Raw BGRA instance-segmentation image for simulation frame ``frame_id``."""
        return self._wait(self._instance_by_frame, frame_id, timeout)

    def spawn_camera(
        self,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float] = (-45.0, 45.0, 0.0),
        with_instance_segmentation: bool = False,
        sensor_tick: float = 0.05,
    ) -> Any:
        """Spawn RGB camera actor at given (X, Y, Z) and (pitch, yaw, roll).

        With ``with_instance_segmentation`` a second, identically configured
        ``sensor.camera.instance_segmentation`` is mounted at the same pose; it
        renders the same ticks and is used only to derive exact labels.

        ``sensor_tick`` is the capture interval in seconds. In synchronous mode
        pass 0.0 to render on every world tick: a tick equal to the world's
        fixed delta lets floating-point drift make sensors skip random ticks.
        """
        if self.world is None or carla is None:
            raise RuntimeError("CARLA world is not available.")

        bp_library = self.world.get_blueprint_library()
        cam_bp = bp_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.width))
        cam_bp.set_attribute("image_size_y", str(self.height))
        cam_bp.set_attribute("fov", str(self.fov))
        cam_bp.set_attribute("sensor_tick", str(sensor_tick))

        loc = carla.Location(x=position[0], y=position[1], z=position[2])
        rot = carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2])
        transform = carla.Transform(loc, rot)

        if self.sensor_actor is not None:
            self.sensor_actor.stop()
            self.sensor_actor.destroy()

        self.sensor_actor = self.world.spawn_actor(cam_bp, transform)

        def _process_image(image: Any) -> None:
            raw = np.frombuffer(image.raw_data, dtype=np.uint8)
            bgra = raw.reshape((image.height, image.width, 4))
            frame_bgr = bgra[:, :, :3].copy()
            self.latest_frame = frame_bgr
            self.frame_count += 1
            self._store(self._rgb_by_frame, image.frame, frame_bgr)
            try:
                self.image_queue.put_nowait((image.frame, frame_bgr))
            except queue.Full:
                try:
                    self.image_queue.get_nowait()
                    self.image_queue.put_nowait((image.frame, frame_bgr))
                except Exception:
                    pass

        self.sensor_actor.listen(_process_image)

        if self.instance_sensor is not None:
            self.instance_sensor.stop()
            self.instance_sensor.destroy()
            self.instance_sensor = None
        if with_instance_segmentation:
            seg_bp = bp_library.find("sensor.camera.instance_segmentation")
            seg_bp.set_attribute("image_size_x", str(self.width))
            seg_bp.set_attribute("image_size_y", str(self.height))
            seg_bp.set_attribute("fov", str(self.fov))
            seg_bp.set_attribute("sensor_tick", str(sensor_tick))
            self.instance_sensor = self.world.spawn_actor(seg_bp, transform)

            def _process_instance(image: Any) -> None:
                raw = np.frombuffer(image.raw_data, dtype=np.uint8)
                bgra = raw.reshape((image.height, image.width, 4)).copy()
                self._store(self._instance_by_frame, image.frame, bgra)

            self.instance_sensor.listen(_process_instance)

        return self.sensor_actor

    def get_frame(self, frame_id: Optional[int] = None, timeout: float = 2.0) -> Optional[np.ndarray]:
        """Fetch frame synchronized to specific simulation tick."""
        if frame_id is None:
            return self.latest_frame
        start_t = time.time() if "time" in globals() else 0
        try:
            while True:
                f_id, frame = self.image_queue.get(timeout=timeout)
                if f_id == frame_id:
                    return frame
                elif f_id > frame_id:
                    return frame
        except Exception:
            return self.latest_frame

    def project_world_to_image(
        self, world_location: Any
    ) -> Optional[Tuple[int, int]]:
        """Project a 3D world location to 2D pixel coordinates (u, v)."""
        if self.sensor_actor is None:
            return None

        # Camera transform matrix
        cam_transform = self.sensor_actor.get_transform()
        world_to_cam = np.array(cam_transform.get_inverse_matrix())

        p_world = np.array([world_location.x, world_location.y, world_location.z, 1.0])
        p_cam = np.dot(world_to_cam, p_world)

        # In Unreal Engine: X forward, Y right, Z up
        # Standard camera frame: X right, Y down, Z forward
        p_cam_std = np.array([p_cam[1], -p_cam[2], p_cam[0]])

        # If behind camera plane, discard
        if p_cam_std[2] <= 0.1:
            return None

        p_proj = np.dot(self.K, p_cam_std)
        u = int(p_proj[0] / p_proj[2])
        v = int(p_proj[1] / p_proj[2])

        if 0 <= u < self.width and 0 <= v < self.height:
            return u, v
        return None

    def compute_2d_bounding_box(
        self, actor: Any, min_size: int = 10
    ) -> Optional[Tuple[int, int, int, int]]:
        """Project 8 3D bounding box corners to image plane to obtain 2D box [x1, y1, x2, y2].

        This is the hull of the 3D box and ignores occlusion; boxes narrower or
        shorter than ``min_size`` pixels are discarded.
        """
        if self.sensor_actor is None:
            return None

        bb = actor.bounding_box
        trans = actor.get_transform()
        corners_3d = bb.get_world_vertices(trans)

        cam_transform = self.sensor_actor.get_transform()
        world_to_cam = np.array(cam_transform.get_inverse_matrix())

        u_coords = []
        v_coords = []

        for pt in corners_3d:
            p_world = np.array([pt.x, pt.y, pt.z, 1.0])
            p_cam = np.dot(world_to_cam, p_world)
            p_cam_std = np.array([p_cam[1], -p_cam[2], p_cam[0]])
            if p_cam_std[2] <= 0.5:  # Behind camera plane
                continue
            p_proj = np.dot(self.K, p_cam_std)
            u = p_proj[0] / p_proj[2]
            v = p_proj[1] / p_proj[2]
            u_coords.append(u)
            v_coords.append(v)

        if len(u_coords) < 4:
            return None

        x1 = max(0, int(np.floor(min(u_coords))))
        y1 = max(0, int(np.floor(min(v_coords))))
        x2 = min(self.width - 1, int(np.ceil(max(u_coords))))
        y2 = min(self.height - 1, int(np.ceil(max(v_coords))))

        if x2 < x1 or y2 < y1 or x2 - x1 < min_size or y2 - y1 < min_size:
            return None

        return int(x1), int(y1), int(x2), int(y2)

    def destroy(self) -> None:
        """Clean up sensor actors."""
        if self.instance_sensor is not None:
            self.instance_sensor.stop()
            self.instance_sensor.destroy()
            self.instance_sensor = None
        if self.sensor_actor is not None:
            self.sensor_actor.stop()
            self.sensor_actor.destroy()
            self.sensor_actor = None

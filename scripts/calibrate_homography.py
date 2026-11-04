"""Calibrate Road-Plane Perspective Homography for Roundabout Camera Rig.

Projects 4 known metric ground points in the roundabout plane (Z=0.3m)
into camera sensor pixels (u, v) using camera intrinsic/extrinsic geometry,
and derives the 3x3 Homography matrix H mapping pixels to metric road coordinates.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.perception.homography import RoadwayHomography


def compute_town03_homography(
    cam_pos: Tuple[float, float, float] = (2.0, -32.0, 16.3),
    cam_rot: Tuple[float, float, float] = (-26.0, 90.0, 0.0),
    image_size: Tuple[int, int] = (1920, 1080),
    fov: float = 85.0,
    roundabout_center: Tuple[float, float, float] = (2.0, 6.0, 0.3),
) -> Tuple[RoadwayHomography, List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Derive ground-plane homography using exact pinhole camera projection."""
    w, h = image_size
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.array([
        [focal, 0.0, w / 2.0],
        [0.0, focal, h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Rotation matrix from pitch, yaw, roll (in CARLA degrees)
    pitch = math.radians(cam_rot[0])
    yaw = math.radians(cam_rot[1])
    roll = math.radians(cam_rot[2])

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # CARLA / Unreal rotation: Rz(yaw) * Ry(pitch) * Rx(roll)
    R_world_to_cam = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ], dtype=np.float64)

    t_cam = np.array(cam_pos, dtype=np.float64)

    def _project_point(wx: float, wy: float, wz: float) -> Tuple[float, float]:
        p_world = np.array([wx, wy, wz], dtype=np.float64)
        p_rel = p_world - t_cam
        p_cam = np.dot(R_world_to_cam.T, p_rel)
        p_cam_std = np.array([p_cam[1], -p_cam[2], p_cam[0]])
        proj = np.dot(K, p_cam_std)
        u = float(proj[0] / proj[2])
        v = float(proj[1] / proj[2])
        return u, v

    # 4 reference points on the roundabout perimeter (Z = 0.3m)
    cx, cy, cz = roundabout_center
    R_ref = 18.0
    ground_points_metric = [
        (cx - R_ref, cy - R_ref),
        (cx + R_ref, cy - R_ref),
        (cx + R_ref, cy + R_ref),
        (cx - R_ref, cy + R_ref),
    ]

    pixel_points = []
    for gx, gy in ground_points_metric:
        u, v = _project_point(gx, gy, cz)
        pixel_points.append((u, v))

    homography = RoadwayHomography(pixel_points, ground_points_metric)

    # Save to config file
    config_data = {
        "cam_pos": list(cam_pos),
        "cam_rot": list(cam_rot),
        "fov": fov,
        "image_size": list(image_size),
        "ground_z": cz,
        "src_pixel_points": pixel_points,
        "dst_road_points": ground_points_metric,
        "H_matrix": homography.H.tolist(),
        "H_inv_matrix": homography.H_inv.tolist(),
    }

    os.makedirs("configs", exist_ok=True)
    with open("configs/homography_town03.json", "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"[Homography] Calibrated 3x3 homography matrix for Town03 camera at h={cam_pos[2]}m.")
    return homography, pixel_points, ground_points_metric


if __name__ == "__main__":
    compute_town03_homography()

"""Perspective Homography & Inverse Perspective Mapping (IPM) for RoundWatch.

Transforms 2D image coordinates (camera pixel space) to 2D metric road-plane
coordinates (meters) and vice-versa for AR corridor projection.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np


class RoadwayHomography:
    """Manages ground-plane homography calibration and coordinate transforms."""

    def __init__(
        self,
        src_points: Optional[List[Tuple[float, float]]] = None,
        dst_points: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Initialize homography matrix.
        
        src_points: 4 pixel coordinates [(u1, v1), (u2, v2), (u3, v3), (u4, v4)]
        dst_points: 4 road-plane metric coordinates [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
        """
        if src_points is not None and dst_points is not None:
            self.calibrate(src_points, dst_points)
        else:
            # Default identity calibration
            self.H = np.eye(3, dtype=np.float64)
            self.H_inv = np.eye(3, dtype=np.float64)

    def calibrate(
        self,
        src_points: List[Tuple[float, float]],
        dst_points: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Compute 3x3 homography matrix H from 4 corresponding point pairs."""
        if len(src_points) != 4 or len(dst_points) != 4:
            raise ValueError("Exactly 4 point pairs are required for homography calibration.")

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)

        self.H, _ = cv2.findHomography(src, dst)
        self.H_inv = np.linalg.inv(self.H)
        return self.H

    def pixel_to_road(self, u: float, v: float) -> Tuple[float, float]:
        """Convert image pixel coordinate (u, v) to metric road-plane coordinate (X, Y)."""
        pt_homogeneous = np.array([u, v, 1.0], dtype=np.float64)
        road_homogeneous = np.dot(self.H, pt_homogeneous)
        
        scale = road_homogeneous[2]
        if abs(scale) < 1e-6:
            scale = 1e-6

        X = float(road_homogeneous[0] / scale)
        Y = float(road_homogeneous[1] / scale)
        return X, Y

    def road_to_pixel(self, X: float, Y: float) -> Tuple[int, int]:
        """Convert metric road-plane coordinate (X, Y) to image pixel coordinate (u, v)."""
        pt_homogeneous = np.array([X, Y, 1.0], dtype=np.float64)
        img_homogeneous = np.dot(self.H_inv, pt_homogeneous)

        scale = img_homogeneous[2]
        if abs(scale) < 1e-6:
            scale = 1e-6

        u = int(round(img_homogeneous[0] / scale))
        v = int(round(img_homogeneous[1] / scale))
        return u, v

    def bbox_bottom_center_to_road(
        self, bbox: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """Extract bottom-center of bounding box (ground contact point) and map to road."""
        x1, y1, x2, y2 = bbox
        u_center = (x1 + x2) / 2.0
        v_bottom = float(y2)
        return self.pixel_to_road(u_center, v_bottom)

    def project_polygon_to_image(
        self, road_polygon: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Project 2D metric polygon corridor to pixel space for AR visual overlay."""
        pixel_points = [self.road_to_pixel(x, y) for x, y in road_polygon]
        return np.array(pixel_points, dtype=np.int32)


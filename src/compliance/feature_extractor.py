"""Trajectory Feature Extractor for Portuguese Highway Code Roundabout Analysis.

Converts variable-length raw tracked vehicle trajectories (x_t, y_t, t) into
fixed-dimensional normalized spatial-temporal feature representations suitable
for AI classification (MLP / 1D-CNN) and rule-based verification.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.compliance.geometry import RoundaboutGeometry


class TrajectoryFeatureExtractor:
    """Extracts normalized spatial and geometric features from roundabout vehicle paths."""

    def __init__(
        self,
        geometry: Optional[RoundaboutGeometry] = None,
        num_resample_points: int = 32,
    ) -> None:
        self.geom = geometry or RoundaboutGeometry()
        self.num_resample_points = num_resample_points

    def resample_path(
        self, points: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Resample a 2D polyline into N equidistant points along arc length."""
        if len(points) == 0:
            return np.zeros((self.num_resample_points, 2), dtype=np.float32)
        if len(points) == 1:
            return np.tile(points[0], (self.num_resample_points, 1)).astype(np.float32)

        pts = np.array(points, dtype=np.float32)
        diffs = pts[1:] - pts[:-1]
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        cum_dist = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_len = cum_dist[-1]

        if total_len < 1e-4:
            return np.tile(pts[0], (self.num_resample_points, 1))

        target_distances = np.linspace(0.0, total_len, self.num_resample_points)
        x_interp = np.interp(target_distances, cum_dist, pts[:, 0])
        y_interp = np.interp(target_distances, cum_dist, pts[:, 1])

        return np.column_stack((x_interp, y_interp))

    def extract_features(
        self,
        trajectory: List[Tuple[float, float]],
        entry_leg: Optional[int] = None,
        exit_leg: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract both fixed-size numerical vector and interpretable dictionary features."""
        if len(trajectory) < 2:
            return {
                "vector": np.zeros(self.num_resample_points * 2 + 10, dtype=np.float32),
                "is_valid": False,
                "metrics": {},
            }

        # 1. Resample and normalize coordinates relative to center and outer radius
        resampled_raw = self.resample_path(trajectory)
        dx = resampled_raw[:, 0] - self.geom.cx
        dy = resampled_raw[:, 1] - self.geom.cy
        norm_coords = np.column_stack((dx / self.geom.r_outer, dy / self.geom.r_outer))

        # 2. Polar transformations
        radii = np.hypot(dx, dy)
        angles_deg = (np.degrees(np.arctan2(dy, dx))) % 360.0

        r_mean = float(np.mean(radii))
        r_min = float(np.min(radii))
        r_max = float(np.max(radii))
        r_std = float(np.std(radii))

        # 3. Zone classifications
        zones = [self.geom.classify_zone(pt[0], pt[1]) for pt in trajectory]
        ring_zones = [z for z in zones if z in ("inner_ring", "outer_ring")]
        total_ring_pts = max(1, len(ring_zones))

        outer_ratio = ring_zones.count("outer_ring") / total_ring_pts
        inner_ratio = ring_zones.count("inner_ring") / total_ring_pts

        # 4. Traversal angles
        entry_angle = float(angles_deg[0])
        exit_angle = float(angles_deg[-1])
        circ_traversal = (exit_angle - entry_angle) % 360.0

        if entry_leg is None:
            entry_leg = self.geom.identify_leg(trajectory[0][0], trajectory[0][1])
        if exit_leg is None:
            exit_leg = self.geom.identify_leg(trajectory[-1][0], trajectory[-1][1])

        num_exits_traversed = 1
        if entry_leg is not None and exit_leg is not None:
            num_exits_traversed = (exit_leg - entry_leg) % self.geom.num_legs
            if num_exits_traversed == 0:
                num_exits_traversed = self.geom.num_legs
        else:
            num_exits_traversed = max(1, int(round(circ_traversal / 90.0)))

        # 5. Lane change transition fraction
        # Find index where vehicle transitions from inner to outer ring
        transition_frac = -1.0
        for i in range(len(zones) - 1):
            if zones[i] == "inner_ring" and zones[i + 1] == "outer_ring":
                transition_frac = i / len(zones)
                break

        # 6. Assemble scalar summary features (10 dimensions)
        scalar_features = np.array([
            r_mean / self.geom.r_outer,
            r_min / self.geom.r_outer,
            r_max / self.geom.r_outer,
            r_std / self.geom.r_outer,
            outer_ratio,
            inner_ratio,
            circ_traversal / 360.0,
            num_exits_traversed / float(self.geom.num_legs),
            transition_frac if transition_frac >= 0 else 0.0,
            1.0 if transition_frac >= 0 else 0.0,
        ], dtype=np.float32)

        # 7. Concatenate normalized coordinates (64 dims) + scalar features (10 dims) = 74 dims
        feature_vector = np.concatenate([norm_coords.flatten(), scalar_features])

        return {
            "vector": feature_vector,
            "is_valid": True,
            "metrics": {
                "num_points": len(trajectory),
                "resampled_coords": resampled_raw.tolist(),
                "entry_angle": round(entry_angle, 1),
                "exit_angle": round(exit_angle, 1),
                "circ_traversal_deg": round(circ_traversal, 1),
                "num_exits": num_exits_traversed,
                "outer_ratio": round(outer_ratio, 3),
                "inner_ratio": round(inner_ratio, 3),
                "transition_progress": round(transition_frac, 3) if transition_frac >= 0 else None,
                "r_mean_m": round(r_mean, 2),
                "r_std_m": round(r_std, 2),
            },
        }


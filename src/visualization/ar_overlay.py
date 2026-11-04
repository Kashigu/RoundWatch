"""Augmented Reality (AR) Overlay Engine for Roundabout Rule Compliance.

Implements dynamic visual projection directly onto 1080p video frames:
- Green Corridor: Permitted / legal circulatory paths per Art. 14.º-A
- Red Corridor: Prohibited maneuvers / detected infractions
- Blue Trail: Actual executed vehicle trajectory
- Azuma AR Principle: Corridors rendered on-demand (per selected vehicle) to eliminate visual clutter.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.compliance.classifier import TrajectoryClassifier
from src.compliance.geometry import RoundaboutGeometry
from src.perception.homography import RoadwayHomography


class AROverlayEngine:
    """Renders semi-transparent AR corridors, compliance badges, and infraction evidence."""

    # Color definitions (BGR format for OpenCV)
    COLOR_LEGAL_GREEN = (40, 205, 50)       # Legal corridor
    COLOR_INFRACTION_RED = (40, 40, 235)     # Infraction / prohibited corridor
    COLOR_TRAIL_BLUE = (255, 140, 0)         # Actual vehicle trail
    COLOR_COMPLIANT_BADGE = (30, 180, 40)
    COLOR_VIOLATION_BADGE = (30, 30, 220)
    COLOR_TEXT = (255, 255, 255)

    def __init__(
        self,
        geometry: Optional[RoundaboutGeometry] = None,
        homography: Optional[RoadwayHomography] = None,
        corridor_width_m: float = 3.6,
        alpha: float = 0.40,
    ) -> None:
        self.geom = geometry or RoundaboutGeometry()
        self.homography = homography or RoadwayHomography()
        self.corridor_width_m = corridor_width_m
        self.alpha = alpha

        # Output directory for infraction evidence
        self.evidence_dir = Path("output/infractions")
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.saved_infractions: set = set()

    def generate_legal_corridor_road(
        self, entry_leg: int, exit_leg: int, num_pts: int = 40
    ) -> List[Tuple[float, float]]:
        """Generate centerline points of legal path per Portuguese Highway Code."""
        cx, cy = self.geom.cx, self.geom.cy
        r_outer = self.geom.r_outer - 1.2
        r_inner = self.geom.r_inner - 1.2

        entry_ang = self.geom.leg_angles[entry_leg]
        num_exits = (exit_leg - entry_leg) % self.geom.num_legs
        if num_exits == 0:
            num_exits = self.geom.num_legs
        span_deg = num_exits * 90.0

        corridor_pts: List[Tuple[float, float]] = []

        if num_exits == 1:
            # Legal path for Exit 1: stay in outer lane
            for i in range(num_pts):
                frac = i / float(num_pts - 1)
                ang = math.radians(entry_ang - frac * span_deg)
                px = cx + r_outer * math.cos(ang)
                py = cy + r_outer * math.sin(ang)
                corridor_pts.append((px, py))
        else:
            # Legal path for Exit 2+: inner ring, move outward after prior exit
            shift_start_deg = (num_exits - 1) * 90.0 + 10.0
            shift_end_deg = shift_start_deg + 40.0

            for i in range(num_pts):
                frac = i / float(num_pts - 1)
                cur_span = frac * span_deg
                ang = math.radians(entry_ang - cur_span)

                if cur_span < shift_start_deg:
                    r = r_inner
                elif cur_span < shift_end_deg:
                    t = (cur_span - shift_start_deg) / (shift_end_deg - shift_start_deg)
                    r = r_inner + t * (r_outer - r_inner)
                else:
                    r = r_outer

                px = cx + r * math.cos(ang)
                py = cy + r * math.sin(ang)
                corridor_pts.append((px, py))

        return corridor_pts

    def generate_corridor_polygon(
        self, centerline: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Create a 2D ribbon polygon by extruding normal offsets along centerline."""
        if len(centerline) < 2:
            return np.zeros((0, 2), dtype=np.int32)

        half_w = self.corridor_width_m / 2.0
        left_pts = []
        right_pts = []

        for i in range(len(centerline)):
            if i < len(centerline) - 1:
                dx = centerline[i + 1][0] - centerline[i][0]
                dy = centerline[i + 1][1] - centerline[i][1]
            else:
                dx = centerline[i][0] - centerline[i - 1][0]
                dy = centerline[i][1] - centerline[i - 1][1]

            length = math.hypot(dx, dy)
            if length < 1e-4:
                length = 1.0
            nx = -dy / length
            ny = dx / length

            lx = centerline[i][0] + nx * half_w
            ly = centerline[i][1] + ny * half_w
            rx = centerline[i][0] - nx * half_w
            ry = centerline[i][1] - ny * half_w

            # Project metric road coordinates to camera pixels
            u_l, v_l = self.homography.road_to_pixel(lx, ly)
            u_r, v_r = self.homography.road_to_pixel(rx, ry)

            left_pts.append((u_l, v_l))
            right_pts.append((u_r, v_r))

        poly_pixels = left_pts + list(reversed(right_pts))
        return np.array(poly_pixels, dtype=np.int32)

    def render_overlay(
        self,
        frame: np.ndarray,
        active_tracks: List[Dict[str, Any]],
        compliance_results: Dict[int, Dict[str, Any]],
        selected_track_id: Optional[int] = None,
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> np.ndarray:
        """Render complete AR scene: badges, corridors on-demand, and HUD."""
        annotated = frame.copy()
        overlay = frame.copy()

        # 1. Render On-Demand Corridors (Azuma AR principle: only for selected vehicle)
        if selected_track_id is not None:
            res = compliance_results.get(selected_track_id, {})
            is_compliant = res.get("is_compliant", True)
            metrics = res.get("metrics", {})

            # Recover entry and exit legs
            entry_x = res.get("entry_x")
            entry_y = res.get("entry_y")
            if entry_x is None or entry_y is None:
                sel_trk = next((t for t in active_tracks if t.get("track_id") == selected_track_id), None)
                if sel_trk and sel_trk.get("history_pixels"):
                    first_px = sel_trk["history_pixels"][0]
                    entry_x, entry_y = self.homography.pixel_to_road(first_px[0], first_px[1])
                else:
                    entry_x, entry_y = 2.0, 27.0

            e_leg = self.geom.identify_leg(entry_x, entry_y)
            num_exits = metrics.get("num_exits", 2)
            x_leg = (e_leg + num_exits) % self.geom.num_legs

            # Legal corridor (Green)
            legal_center = self.generate_legal_corridor_road(e_leg, x_leg)
            legal_poly = self.generate_corridor_polygon(legal_center)
            if len(legal_poly) > 2:
                cv2.fillPoly(overlay, [legal_poly], self.COLOR_LEGAL_GREEN)

            # If vehicle committed infraction, highlight executed path in Red corridor
            if not is_compliant:
                executed_pts = [(pt[0], pt[1]) for pt in res.get("raw_trajectory", [])]
                if not executed_pts:
                    sel_trk = next((t for t in active_tracks if t.get("track_id") == selected_track_id), None)
                    if sel_trk and sel_trk.get("history_pixels"):
                        executed_pts = [self.homography.pixel_to_road(p[0], p[1]) for p in sel_trk["history_pixels"]]
                if len(executed_pts) >= 2:
                    infraction_poly = self.generate_corridor_polygon(executed_pts)
                    if len(infraction_poly) > 2:
                        cv2.fillPoly(overlay, [infraction_poly], self.COLOR_INFRACTION_RED)

            # Blend transparent overlay with base frame
            cv2.addWeighted(overlay, self.alpha, annotated, 1.0 - self.alpha, 0, annotated)

        # 2. Render Vehicle Bounding Boxes (Clean Object Detection Mode)
        for trk in active_tracks:
            tid = trk.get("track_id", 0)
            bbox = trk["bbox"]
            conf = trk.get("conf", 0.0)
            cls_name = trk.get("class_name", "car")
            is_selected = (selected_track_id == tid)

            # High-visibility detection styling (Green by default, Cyan if clicked/selected)
            box_color = (0, 230, 70) if not is_selected else (0, 220, 255)
            box_thickness = 3 if is_selected else 2

            # Vehicle Bounding Box
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, box_thickness)

            # Label Pill: Vehicle Class + Persistent Tracking ID (e.g., "car #1", "truck #3")
            tag_text = f"{cls_name} #{tid}"

            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(
                annotated,
                (bbox[0], max(0, bbox[1] - th - 8)),
                (bbox[0] + tw + 8, max(th + 8, bbox[1])),
                box_color,
                -1,
            )
            cv2.putText(
                annotated,
                tag_text,
                (bbox[0] + 4, max(th + 3, bbox[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )

            # Draw trajectory trail only if vehicle is explicitly clicked by user
            if is_selected:
                trail = trk.get("history_pixels", [])
                for k in range(1, len(trail)):
                    cv2.line(annotated, trail[k - 1], trail[k], self.COLOR_TRAIL_BLUE, 2)

        # 3. HUD System Status Banner (Clean Detection Header)
        cv2.rectangle(annotated, (20, 20), (520, 80), (20, 20, 20), -1)
        cv2.rectangle(annotated, (20, 20), (520, 80), (70, 70, 70), 1)
        cv2.putText(
            annotated,
            f"RoundWatch Surveillance | Frame: {frame_idx:04d} ({timestamp:.2f}s)",
            (35, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
        )
        mode_text = f"Detected Vehicles: {len(active_tracks)} | YOLOv8 Detection Mode"
        cv2.putText(
            annotated,
            mode_text,
            (35, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
        )

        return annotated

    def _save_evidence_snapshot(
        self,
        frame: np.ndarray,
        tid: int,
        trk: Dict[str, Any],
        res: Dict[str, Any],
        timestamp: float,
    ) -> None:
        """Capture and persist timestamped infraction evidence image."""
        snap = frame.copy()
        bbox = trk["bbox"]
        cv2.rectangle(snap, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.COLOR_INFRACTION_RED, 3)

        citation = res.get("citation", "Infração ao Art. 14.º-A")
        reason = res.get("reason", "Violação de vias de circulação em rotunda.")

        # Overlay evidence banner at bottom
        h, w = snap.shape[:2]
        cv2.rectangle(snap, (0, h - 90), (w, h), (0, 0, 0), -1)
        cv2.putText(
            snap,
            f"[INFRACTION FLAGGED] Vehicle ID: {tid} | Time: {timestamp:.2f}s | {citation}",
            (30, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            snap,
            reason,
            (30, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )

        filename = self.evidence_dir / f"infraction_id{tid}_t{int(timestamp*100):05d}.jpg"
        cv2.imwrite(str(filename), snap)
        print(f"[AR Overlay] Infraction evidence recorded: {filename}")


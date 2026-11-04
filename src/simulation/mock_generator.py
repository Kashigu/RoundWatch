"""Synthetic Mock Generator for RoundWatch.

Generates ground-truth trajectories and simulated elevated camera feeds
matching CARLA Town03 roundabout geometry and Portuguese Highway Code rules.

Key Characteristics:
- Full entry-to-exit transits for every vehicle (starts at approach arm, exits at departure arm)
- Natural tangential entry into the circulatory stream
- Strict Portuguese Highway Code (Art. 14.º-A) lane changes after the prior exit
- Collision-free car-following and distance maintenance
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class MockRoundaboutGenerator:
    """Generates continuous 3-phase vehicle trajectories (Entry -> Circulate -> Exit)."""

    def __init__(
        self,
        fps: int = 20,
        duration_sec: float = 32.0,
        frame_size: Tuple[int, int] = (1920, 1080),
        seed: int = 42,
    ) -> None:
        self.fps = fps
        self.duration_sec = duration_sec
        self.num_frames = int(fps * duration_sec)
        self.width, self.height = frame_size
        self.dt = 1.0 / fps
        np.random.seed(seed)

        # Roundabout geometry (meters)
        self.r_island = 13.5
        self.r_inner = 16.0
        self.r_outer = 19.5
        self.r_limit = 22.0
        self.arm_distance = 38.0

        # Scale factor mapping meters [-38, 38] to 1080p canvas
        self.pixels_per_meter = min(self.width, self.height) / 80.0
        self.origin_px = (self.width // 2, self.height // 2)

        self.MANEUVER_TYPES = [
            "Compliant_Exit1",
            "Compliant_Exit2Plus",
            "Infraction_OuterLaneMisuse",
            "Infraction_DirectCutOffExit",
        ]

        self.VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

    def road_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert metric road-plane coordinate to pixel coordinates (top-down view)."""
        u = int(self.origin_px[0] + x * self.pixels_per_meter)
        v = int(self.origin_px[1] - y * self.pixels_per_meter)
        return u, v

    def pixel_to_road(self, u: float, v: float) -> Tuple[float, float]:
        """Convert pixel coordinates back to metric road-plane coordinates."""
        x = (u - self.origin_px[0]) / self.pixels_per_meter
        y = (self.origin_px[1] - v) / self.pixels_per_meter
        return float(x), float(y)

    def _generate_canonical_waypoints(
        self, maneuver: str, exit_offset: int
    ) -> List[Tuple[float, float, str]]:
        """Generate waypoints assuming entry from South (Leg 1)."""
        waypoints: List[Tuple[float, float, str]] = []
        is_outer_entry = maneuver in ["Compliant_Exit1", "Infraction_OuterLaneMisuse"]

        # 1. Approach straight along South arm
        lat_entry = 5.0 if is_outer_entry else 1.8
        lane_approach = "entry_outer" if is_outer_entry else "entry_inner"

        for y in np.linspace(-self.arm_distance, -20.5, 16):
            waypoints.append((lat_entry, float(y), lane_approach))

        # 2. Smooth entry curve into circulatory flow (turning counter-clockwise)
        th_entry = math.radians(-75.0)

        if is_outer_entry:
            x_target = self.r_outer * math.cos(th_entry)
            y_target = self.r_outer * math.sin(th_entry)
            for f in np.linspace(0.2, 1.0, 6):
                wx = lat_entry + f * (x_target - lat_entry)
                wy = -20.5 + f * (y_target - (-20.5))
                waypoints.append((wx, wy, "entry_merge_outer"))
        else:
            x_target = self.r_inner * math.cos(th_entry)
            y_target = self.r_inner * math.sin(th_entry)
            for f in np.linspace(0.2, 1.0, 6):
                wx = lat_entry + f * (x_target - lat_entry)
                wy = -20.5 + f * (y_target - (-20.5))
                waypoints.append((wx, wy, "entry_merge_inner"))

        # 3. Circulatory Navigation with Highway Code Lane Discipline
        exit_angles = {
            1: math.radians(-11.8),
            2: math.radians(78.2),
            3: math.radians(168.2),
        }
        target_exit_angle = exit_angles[exit_offset]

        if maneuver == "Compliant_Exit1":
            for th in np.linspace(th_entry, target_exit_angle, 20):
                wx = self.r_outer * math.cos(th)
                wy = self.r_outer * math.sin(th)
                waypoints.append((wx, wy, "outer"))

        elif maneuver == "Compliant_Exit2Plus":
            prior_exit_angle = exit_angles[exit_offset - 1]

            # In inner ring up to prior exit
            for th in np.linspace(th_entry, prior_exit_angle, 16 * exit_offset):
                wx = self.r_inner * math.cos(th)
                wy = self.r_inner * math.sin(th)
                waypoints.append((wx, wy, "inner"))

            # Shift to outer ring immediately after prior exit
            th_shift_start = prior_exit_angle + math.radians(6.0)
            th_shift_end = prior_exit_angle + math.radians(40.0)

            for th in np.linspace(th_shift_start, th_shift_end, 14):
                f = (th - th_shift_start) / (th_shift_end - th_shift_start)
                r = self.r_inner + f * (self.r_outer - self.r_inner)
                wx = r * math.cos(th)
                wy = r * math.sin(th)
                waypoints.append((wx, wy, "lane_change_to_outer"))

            # Circulate in outer ring approaching exit
            for th in np.linspace(th_shift_end, target_exit_angle, 14):
                wx = self.r_outer * math.cos(th)
                wy = self.r_outer * math.sin(th)
                waypoints.append((wx, wy, "outer"))

        elif maneuver == "Infraction_OuterLaneMisuse":
            for th in np.linspace(th_entry, target_exit_angle, 28 * exit_offset):
                wx = self.r_outer * math.cos(th)
                wy = self.r_outer * math.sin(th)
                waypoints.append((wx, wy, "outer_misuse"))

        elif maneuver == "Infraction_DirectCutOffExit":
            th_cut_start = target_exit_angle - math.radians(10.0)
            for th in np.linspace(th_entry, th_cut_start, 24 * exit_offset):
                wx = self.r_inner * math.cos(th)
                wy = self.r_inner * math.sin(th)
                waypoints.append((wx, wy, "inner"))

            for th in np.linspace(th_cut_start, target_exit_angle, 8):
                f = (th - th_cut_start) / (target_exit_angle - th_cut_start)
                r = self.r_inner + f * (self.r_outer - self.r_inner)
                wx = r * math.cos(th)
                wy = r * math.sin(th)
                waypoints.append((wx, wy, "direct_inner_cutoff"))

        # 4. Departure straight along designated exit arm
        if exit_offset == 1:
            for x in np.linspace(19.08, self.arm_distance, 16):
                waypoints.append((float(x), -4.0, "exit_lane"))
        elif exit_offset == 2:
            for y in np.linspace(19.08, self.arm_distance, 16):
                waypoints.append((4.0, float(y), "exit_lane"))
        elif exit_offset == 3:
            for x in np.linspace(-19.08, -self.arm_distance, 16):
                waypoints.append((float(x), 4.0, "exit_lane"))

        return waypoints

    def generate_single_trajectory(
        self, vehicle_id: int, entry_leg: int, maneuver: str, start_time: float
    ) -> Dict[str, Any]:
        """Convenience method to generate an isolated trajectory."""
        exit_offset = 1 if maneuver == "Compliant_Exit1" else 2
        exit_leg = ((entry_leg - 1 + exit_offset) % 4) + 1
        canonical_pts = self._generate_canonical_waypoints(maneuver, exit_offset)
        phi = (entry_leg - 1) * (math.pi / 2.0)
        cos_p, sin_p = math.cos(phi), math.sin(phi)
        rotated_wpts = [
            (cx * cos_p - cy * sin_p, cx * sin_p + cy * cos_p, lane)
            for cx, cy, lane in canonical_pts
        ]

        dists = [0.0]
        for i in range(1, len(rotated_wpts)):
            dists.append(dists[-1] + math.hypot(rotated_wpts[i][0] - rotated_wpts[i-1][0], rotated_wpts[i][1] - rotated_wpts[i-1][1]))
        total_dist = dists[-1]

        trajectory = []
        s = 0.0
        v = 7.5
        step_idx = int(start_time * self.fps)

        while s < total_dist:
            t = step_idx * self.dt
            idx = int(np.searchsorted(dists, s))
            idx = max(1, min(len(rotated_wpts) - 1, idx))
            seg_len = dists[idx] - dists[idx - 1]
            frac = 0.0 if seg_len < 1e-4 else (s - dists[idx - 1]) / seg_len
            x = rotated_wpts[idx - 1][0] + frac * (rotated_wpts[idx][0] - rotated_wpts[idx - 1][0])
            y = rotated_wpts[idx - 1][1] + frac * (rotated_wpts[idx][1] - rotated_wpts[idx - 1][1])
            lane = rotated_wpts[idx][2] if frac > 0.5 else rotated_wpts[idx - 1][2]
            heading = math.atan2(rotated_wpts[idx][1] - rotated_wpts[idx - 1][1], rotated_wpts[idx][0] - rotated_wpts[idx - 1][0])

            u, v_px = self.road_to_pixel(x, y)
            len_px = int(4.5 * self.pixels_per_meter)
            wid_px = int(2.0 * self.pixels_per_meter)
            cos_h, sin_h = abs(math.cos(heading)), abs(math.sin(heading))
            bbox_w = int(len_px * cos_h + wid_px * sin_h)
            bbox_h = int(len_px * sin_h + wid_px * cos_h)

            trajectory.append({
                "frame_idx": step_idx,
                "timestamp": round(t, 3),
                "x": round(x, 3),
                "y": round(y, 3),
                "z": 0.0,
                "heading_deg": round(math.degrees(heading), 1),
                "lane": lane,
                "speed_kmh": round(v * 3.6, 1),
                "bbox": [u - bbox_w // 2, v_px - bbox_h // 2, u + bbox_w // 2, v_px + bbox_h // 2],
            })
            s += v * self.dt
            step_idx += 1

        return {
            "vehicle_id": vehicle_id,
            "vehicle_class": "car",
            "maneuver": maneuver,
            "entry_leg": entry_leg,
            "exit_leg": exit_leg,
            "is_compliant": maneuver.startswith("Compliant"),
            "trajectory": trajectory,
        }

    def generate_dataset(
        self, num_vehicles: int = 16, infraction_ratio: float = 0.35
    ) -> List[Dict[str, Any]]:
        """Generate multi-agent dataset with complete entry-to-exit transits for all vehicles."""
        vehicles_plan = []
        current_t = 0.5
        legs = [1, 2, 3, 4]

        for vid in range(1, num_vehicles + 1):
            entry_leg = legs[(vid - 1) % 4]

            if vid % 3 == 0:
                maneuver = "Infraction_OuterLaneMisuse" if (vid % 2 == 0) else "Infraction_DirectCutOffExit"
                exit_offset = 2
            elif vid % 2 == 0:
                maneuver = "Compliant_Exit1"
                exit_offset = 1
            else:
                maneuver = "Compliant_Exit2Plus"
                exit_offset = int(np.random.choice([2, 3]))

            exit_leg = ((entry_leg - 1 + exit_offset) % 4) + 1
            v_class = np.random.choice(self.VEHICLE_CLASSES, p=[0.75, 0.10, 0.10, 0.05])
            dimensions = {
                "car": (4.5, 2.0),
                "bus": (10.5, 2.5),
                "truck": (8.8, 2.4),
                "motorcycle": (2.2, 0.9),
            }[v_class]

            canonical_pts = self._generate_canonical_waypoints(maneuver, exit_offset)
            phi = (entry_leg - 1) * (math.pi / 2.0)
            cos_p, sin_p = math.cos(phi), math.sin(phi)
            rotated_wpts = [
                (cx * cos_p - cy * sin_p, cx * sin_p + cy * cos_p, lane)
                for cx, cy, lane in canonical_pts
            ]

            dists = [0.0]
            for i in range(1, len(rotated_wpts)):
                dists.append(dists[-1] + math.hypot(rotated_wpts[i][0] - rotated_wpts[i-1][0], rotated_wpts[i][1] - rotated_wpts[i-1][1]))

            vehicles_plan.append({
                "vid": vid,
                "v_class": v_class,
                "maneuver": maneuver,
                "entry_leg": entry_leg,
                "exit_leg": exit_leg,
                "is_compliant": maneuver.startswith("Compliant"),
                "dimensions": dimensions,
                "waypoints": rotated_wpts,
                "dists": dists,
                "total_dist": dists[-1],
                "spawn_time": current_t,
                "s": 0.0,
                "v": 7.5,
                "v0": float(np.random.uniform(7.3, 7.8)),
                "active": False,
                "finished": False,
                "trajectory": [],
            })

            # Stagger entry spawns cleanly
            current_t += 1.4

        # Run step-by-step multi-agent simulation
        for step in range(self.num_frames):
            t = step * self.dt

            # Spawn check
            for vp in vehicles_plan:
                if not vp["active"] and not vp["finished"] and t >= vp["spawn_time"]:
                    vp["active"] = True
                    vp["s"] = 0.0

            active_v = [v for v in vehicles_plan if v["active"]]

            for vp in active_v:
                s = vp["s"]
                dists = vp["dists"]
                wpts = vp["waypoints"]
                idx = int(np.searchsorted(dists, s))
                idx = max(1, min(len(wpts) - 1, idx))
                seg_len = dists[idx] - dists[idx - 1]
                frac = 0.0 if seg_len < 1e-4 else (s - dists[idx - 1]) / seg_len

                x = wpts[idx - 1][0] + frac * (wpts[idx][0] - wpts[idx - 1][0])
                y = wpts[idx - 1][1] + frac * (wpts[idx][1] - wpts[idx - 1][1])
                lane = wpts[idx][2] if frac > 0.5 else wpts[idx - 1][2]
                heading = math.atan2(wpts[idx][1] - wpts[idx - 1][1], wpts[idx][0] - wpts[idx - 1][0])

                # Smooth car-following distance check
                v_target = vp["v0"]
                for other in active_v:
                    if other["vid"] == vp["vid"]:
                        continue
                    ox = other.get("cur_x", 999.0)
                    oy = other.get("cur_y", 999.0)
                    d = math.hypot(ox - x, oy - y)
                    if d < 12.0:
                        dx = ox - x
                        dy = oy - y
                        d_long = dx * math.cos(heading) + dy * math.sin(heading)
                        d_lat = abs(-dx * math.sin(heading) + dy * math.cos(heading))
                        if d_long > 0.5 and d_lat < 2.5:
                            if d_long < 5.0:
                                v_target = min(v_target, 3.5)
                            elif d_long < 8.0:
                                v_target = min(v_target, 5.2)

                vp["cur_x"] = x
                vp["cur_y"] = y
                vp["v"] = v_target
                vp["s"] += vp["v"] * self.dt

                # Pixel bounding box
                u, v_px = self.road_to_pixel(x, y)
                dim_l, dim_w = vp["dimensions"]
                len_px = int(dim_l * self.pixels_per_meter)
                wid_px = int(dim_w * self.pixels_per_meter)
                cos_h = abs(math.cos(heading))
                sin_h = abs(math.sin(heading))
                bbox_w = int(len_px * cos_h + wid_px * sin_h)
                bbox_h = int(len_px * sin_h + wid_px * cos_h)

                vp["trajectory"].append({
                    "frame_idx": step,
                    "timestamp": round(t, 3),
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": 0.0,
                    "heading_deg": round(math.degrees(heading), 1),
                    "lane": lane,
                    "speed_kmh": round(vp["v"] * 3.6, 1),
                    "bbox": [u - bbox_w // 2, v_px - bbox_h // 2, u + bbox_w // 2, v_px + bbox_h // 2],
                })

                if vp["s"] >= vp["total_dist"]:
                    vp["active"] = False
                    vp["finished"] = True

        dataset = []
        for vp in vehicles_plan:
            dataset.append({
                "vehicle_id": vp["vid"],
                "vehicle_class": vp["v_class"],
                "maneuver": vp["maneuver"],
                "entry_leg": vp["entry_leg"],
                "exit_leg": vp["exit_leg"],
                "is_compliant": vp["is_compliant"],
                "trajectory": vp["trajectory"],
            })

        return dataset

    def render_simulation_video(
        self,
        vehicles: List[Dict[str, Any]],
        output_video_path: Path,
        render_ground_truth_overlay: bool = True,
    ) -> None:
        """Render 2D synthetic camera video of the roundabout."""
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path), fourcc, self.fps, (self.width, self.height)
        )

        frame_lookup: Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {
            f: [] for f in range(self.num_frames)
        }
        for veh in vehicles:
            for pt in veh["trajectory"]:
                f_idx = pt["frame_idx"]
                if 0 <= f_idx < self.num_frames:
                    frame_lookup[f_idx].append((veh, pt))

        bg_template = self._create_roadway_template()

        for f_idx in range(self.num_frames):
            frame = bg_template.copy()
            active_vehicles = frame_lookup.get(f_idx, [])

            for veh, pt in active_vehicles:
                x1, y1, x2, y2 = pt["bbox"]
                v_class = veh["vehicle_class"]
                is_comp = veh["is_compliant"]

                color = (40, 200, 40) if is_comp else (40, 40, 230)
                if not render_ground_truth_overlay:
                    color = (200, 200, 200)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                label = f"V{veh['vehicle_id']} {v_class}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            time_sec = f_idx * self.dt
            hud_text = f"Frame: {f_idx:04d} | Time: {time_sec:05.2f}s | Active Vehicles: {len(active_vehicles)}"
            cv2.putText(
                frame,
                hud_text,
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)

        writer.release()

    def _create_roadway_template(self) -> np.ndarray:
        """Create static road canvas with complete approaches, exit lanes, and roundabout markings."""
        canvas = np.full((self.height, self.width, 3), 35, dtype=np.uint8)
        u0, v0 = self.origin_px

        arm_width_px = int(14.0 * self.pixels_per_meter)
        r_lim_px = int(self.r_limit * self.pixels_per_meter)
        r_isl_px = int(self.r_island * self.pixels_per_meter)

        # 1. Road channel asphalt
        cv2.rectangle(canvas, (u0 - arm_width_px // 2, v0), (u0 + arm_width_px // 2, self.height), (60, 60, 60), -1)
        cv2.rectangle(canvas, (u0 - arm_width_px // 2, 0), (u0 + arm_width_px // 2, v0), (60, 60, 60), -1)
        cv2.rectangle(canvas, (u0, v0 - arm_width_px // 2), (self.width, v0 + arm_width_px // 2), (60, 60, 60), -1)
        cv2.rectangle(canvas, (0, v0 - arm_width_px // 2), (u0, v0 + arm_width_px // 2), (60, 60, 60), -1)

        # 2. Circulatory ring asphalt
        cv2.circle(canvas, (u0, v0), r_lim_px, (60, 60, 60), -1)

        # 3. Outer curb
        cv2.circle(canvas, (u0, v0), r_lim_px, (190, 190, 190), 3)

        # 4. Central island
        cv2.circle(canvas, (u0, v0), r_isl_px, (30, 90, 30), -1)
        cv2.circle(canvas, (u0, v0), r_isl_px, (220, 220, 220), 3)

        # 5. Circulatory lane divider (dashed circle between inner and outer ring at 17.8m)
        r_div_px = int(17.8 * self.pixels_per_meter)
        for deg in range(0, 360, 10):
            rad1 = math.radians(deg)
            rad2 = math.radians(deg + 5)
            p1 = (int(u0 + r_div_px * math.cos(rad1)), int(v0 - r_div_px * math.sin(rad1)))
            p2 = (int(u0 + r_div_px * math.cos(rad2)), int(v0 - r_div_px * math.sin(rad2)))
            cv2.line(canvas, p1, p2, (255, 255, 255), 2)

        # 6. Yield lines (dashed lines at the 4 approach mouths)
        y_entry_s = v0 + r_lim_px
        cv2.line(canvas, (u0, y_entry_s), (u0 + arm_width_px // 2, y_entry_s), (255, 255, 255), 4)

        y_entry_n = v0 - r_lim_px
        cv2.line(canvas, (u0 - arm_width_px // 2, y_entry_n), (u0, y_entry_n), (255, 255, 255), 4)

        x_entry_e = u0 + r_lim_px
        cv2.line(canvas, (x_entry_e, v0 - arm_width_px // 2), (x_entry_e, v0), (255, 255, 255), 4)

        x_entry_w = u0 - r_lim_px
        cv2.line(canvas, (x_entry_w, v0), (x_entry_w, v0 + arm_width_px // 2), (255, 255, 255), 4)

        return canvas


def run_generator(
    output_dir: str = "data/mock",
    num_vehicles: int = 16,
    duration_sec: float = 32.0,
    fps: int = 20,
) -> Path:
    """CLI / programmatic entrypoint for running mock generation."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generator = MockRoundaboutGenerator(fps=fps, duration_sec=duration_sec)
    dataset = generator.generate_dataset(num_vehicles=num_vehicles)

    json_file = out_path / "ground_truth_trajectories.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    video_file = out_path / "synthetic_roundabout_feed.mp4"
    generator.render_simulation_video(dataset, video_file)

    metadata = {
        "num_vehicles": len(dataset),
        "duration_sec": duration_sec,
        "fps": fps,
        "classes": generator.VEHICLE_CLASSES,
        "maneuvers": generator.MANEUVER_TYPES,
        "pixels_per_meter": generator.pixels_per_meter,
        "origin_px": generator.origin_px,
        "video_file": str(video_file),
        "json_file": str(json_file),
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out_path


if __name__ == "__main__":
    out = run_generator()
    print(f"Mock dataset generated successfully at {out}")

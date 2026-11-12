"""Multi-Object Tracking (MOT) for RoundWatch, built on the official BoT-SORT tracker.

Per-camera tracking uses the Ultralytics BoT-SORT implementation (Kalman filter, two-stage
BYTE association, GMC camera-motion compensation and optional ReID):
  - ``VehicleDetector.track()`` runs YOLO + BoT-SORT in one step with native ReID features
    (the main path used by the pipeline and the live monitor).
  - ``BoTSORTTracker`` feeds pre-computed detections from any source into BoT-SORT.

``DynamicMultiCameraTracker`` sits on top of the two per-camera BoT-SORT streams (South and
North) and fuses their local track IDs into one global vehicle ID in road coordinates, so a
vehicle keeps its ID across cameras and through the central-island occlusion.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace

DEFAULT_BOTSORT_CFG = "configs/botsort.yaml"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class _DetectionArray:
    """Minimal NumPy-backed stand-in for Ultralytics ``Boxes`` (what BoT-SORT's ``update`` expects)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy.reshape(-1, 4).astype(np.float32)
        self.conf = conf.reshape(-1).astype(np.float32)
        self.cls = cls.reshape(-1).astype(np.float32)

    @property
    def xywh(self) -> np.ndarray:
        out = self.xyxy.copy()
        out[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        out[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        out[:, 0] = self.xyxy[:, 0] + out[:, 2] / 2.0
        out[:, 1] = self.xyxy[:, 1] + out[:, 3] / 2.0
        return out

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx: Any) -> "_DetectionArray":
        return _DetectionArray(self.xyxy[idx], self.conf[idx], self.cls[idx])


def load_botsort_args(config_path: str = DEFAULT_BOTSORT_CFG, **overrides: Any) -> IterableSimpleNamespace:
    """Load a BoT-SORT YAML config (Ultralytics format) and apply keyword overrides."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    return IterableSimpleNamespace(**cfg)


class BoTSORTTracker:
    """Adapter that runs the official Ultralytics BoT-SORT on pre-computed detection dicts.

    Pass the camera frame to ``update`` to enable GMC camera-motion compensation. Native ReID
    (``model: auto``) needs YOLO backbone features, which only ``VehicleDetector.track()`` provides;
    here ReID is active only if the config names an external ReID model.
    """

    def __init__(self, config_path: str = DEFAULT_BOTSORT_CFG, **overrides: Any) -> None:
        args = load_botsort_args(config_path, **overrides)
        if args.with_reid and getattr(args, "model", "auto") == "auto":
            args.with_reid = False  # no detector features available on this path
        self.args = args
        self.tracker = BOTSORT(args)

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Associate this frame's detections and return confirmed tracks with persistent ``track_id``."""
        if detections:
            dets = _DetectionArray(
                np.array([d["bbox"] for d in detections], dtype=np.float32),
                np.array([d["conf"] for d in detections], dtype=np.float32),
                np.array([d.get("class_id", 2) for d in detections], dtype=np.float32),
            )
        else:
            dets = _DetectionArray(np.zeros((0, 4)), np.zeros(0), np.zeros(0))

        out = self.tracker.update(dets, frame)
        tracks: List[Dict[str, Any]] = []
        for row in out:
            x1, y1, x2, y2, tid, score, cls, _idx = row[:8]
            cls_id = int(cls)
            tracks.append({
                "track_id": int(tid),
                "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                "class_id": cls_id,
                "class_name": VEHICLE_CLASSES.get(cls_id, "car"),
                "conf": round(float(score), 3),
            })
        return tracks

    def reset(self) -> None:
        self.tracker.reset()


def pixel_to_road_ray(
    u: float,
    v: float,
    cam_pos: Tuple[float, float, float] | List[float],
    cam_rot: Tuple[float, float, float] | List[float],
    fov: float = 90.0,
    w: int = 1280,
    h: int = 720,
    ground_z: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """Project a 2D camera viewport pixel (u, v) to a 3D road coordinate (X, Y) on plane ground_z.

    Uses exact ray-plane intersection with camera extrinsics (cam_pos, cam_rot).
    """
    focal = (w / 2.0) / math.tan(math.radians(fov / 2.0))
    x_cam = (u - (w / 2.0)) / focal
    y_cam = (v - (h / 2.0)) / focal
    z_cam = 1.0

    v_c = np.array([z_cam, x_cam, -y_cam], dtype=np.float64)
    v_c /= np.linalg.norm(v_c)

    pitch = math.radians(cam_rot[0])
    yaw = math.radians(cam_rot[1])

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    R_w2c = R_pitch @ R_yaw
    R_c2w = R_w2c.T

    ray_w = R_c2w @ v_c
    if abs(ray_w[2]) < 1e-4 or ray_w[2] >= 0:
        return None

    t = (ground_z - cam_pos[2]) / ray_w[2]
    if t <= 0:
        return None

    return float(cam_pos[0] + t * ray_w[0]), float(cam_pos[1] + t * ray_w[1])


class DynamicMultiCameraTracker:
    """Cross-camera identity fusion on top of per-camera BoT-SORT tracks.

    Input per frame: the confirmed BoT-SORT tracks of the South and North cameras, each carrying a
    camera-local ``track_id``. Each (camera, local id) pair is bound to one global vehicle ID:
      - a bound pair keeps its global ID (BoT-SORT already did the temporal association);
      - a new local track is linked, by road-plane distance, to a global vehicle currently seen by
        the other camera (same car in both views) or recently lost (camera hand-over, occlusion
        behind the central island, or a BoT-SORT ID restart); otherwise a new vehicle is created.
    Global IDs are assigned sequentially (1, 2, 3...) once a vehicle reaches ``min_hits`` frames.
    """

    SAME_FRAME_GATE_M = 6.5     # same car seen by both cameras in the same frame
    RECENT_LOST_GATE_M = 8.0    # re-link to a vehicle lost <= 5 frames ago
    LONG_LOST_GATE_M = 14.0     # re-link to a vehicle lost longer (central-island occlusion)
    MERGE_DIST_M = 2.5          # two global vehicles this close ...
    MERGE_FRAMES = 4            # ... for this many consecutive frames are one vehicle (duplicate)

    def __init__(
        self,
        cam_pos_s: List[float] | Tuple[float, float, float],
        cam_rot_s: List[float] | Tuple[float, float, float],
        cam_pos_n: List[float] | Tuple[float, float, float],
        cam_rot_n: List[float] | Tuple[float, float, float],
        max_age_frames: int = 45,
        min_hits: int = 3,
        roi_center: Tuple[float, float] = (2.0, 6.0),
        roi_radius: float = 38.0,
        ground_point: str = "box75",
        merge_duplicates: bool = True,
    ) -> None:
        self.cams = {
            "south": (cam_pos_s, cam_rot_s),
            "north": (cam_pos_n, cam_rot_n),
        }
        self.max_age = max_age_frames
        self.min_hits = min_hits
        self.roi_center = roi_center      # vehicles are only created inside the monitored area
        self.roi_radius = roi_radius
        self.ground_point = ground_point
        self.merge_duplicates = merge_duplicates
        self.next_id = 1  # next public (confirmed) vehicle ID

        self._next_key = 1
        self._vehicles: Dict[int, Dict[str, Any]] = {}       # internal key -> vehicle record
        self._local_to_key: Dict[Tuple[str, int], int] = {}  # (camera, BoT-SORT id) -> internal key
        self._close_pairs: Dict[Tuple[int, int], int] = {}   # (key, key) -> consecutive close frames

        self.confirmed_tracks: List[Dict[str, Any]] = []
        self.completed_tracks: Dict[str, Dict[str, Any]] = {}

    def update(
        self,
        tracks_s: List[Dict[str, Any]],
        tracks_n: List[Dict[str, Any]],
        frame_idx: int,
        dt: float = 0.05,
        pane_w: int = 1280,
        height: int = 720,
    ) -> List[Dict[str, Any]]:
        """Fuse this frame's per-camera BoT-SORT tracks; return confirmed vehicles visible now."""
        obs_by_key: Dict[int, Dict[str, Any]] = {}  # key -> {"south": obs, "north": obs}

        for cam, cam_tracks in (("south", tracks_s), ("north", tracks_n)):
            cam_pos, cam_rot = self.cams[cam]
            unbound: List[Dict[str, Any]] = []

            for t in cam_tracks:
                bx = t["bbox"]
                pt = self._road_point(bx, cam_pos, cam_rot, pane_w, height)
                if pt is None:
                    continue
                obs = {"cam": cam, "local_id": int(t["track_id"]), "bbox": tuple(bx), "road": pt,
                       "class_name": t.get("class_name", "car"), "conf": float(t.get("conf", 0.8))}
                key = self._local_to_key.get((cam, obs["local_id"]))
                if key is not None and key in self._vehicles and cam not in obs_by_key.get(key, {}):
                    obs_by_key.setdefault(key, {})[cam] = obs
                else:
                    unbound.append(obs)

            self._bind_new_local_tracks(cam, unbound, obs_by_key, frame_idx, dt)

        for key, cam_obs in obs_by_key.items():
            self._apply_observation(self._vehicles[key], cam_obs, frame_idx, dt, pane_w)

        for key, veh in self._vehicles.items():
            if key not in obs_by_key:
                veh["active_s"] = None
                veh["active_n"] = None

        if self.merge_duplicates:
            self._merge_duplicates(obs_by_key)
        self._expire(frame_idx)

        self.confirmed_tracks = [v for v in self._vehicles.values() if v["id"] is not None]
        return [v for v in self.confirmed_tracks if v["active_s"] is not None or v["active_n"] is not None]

    def _bind_new_local_tracks(
        self,
        cam: str,
        unbound: List[Dict[str, Any]],
        obs_by_key: Dict[int, Dict[str, Any]],
        frame_idx: int,
        dt: float,
    ) -> None:
        # Candidate (distance, obs index, key) pairs, matched greedily by smallest distance.
        pairs: List[Tuple[float, int, int]] = []
        for oi, obs in enumerate(unbound):
            for key, veh in self._vehicles.items():
                seen_now = obs_by_key.get(key, {})
                if cam in seen_now:
                    continue  # this camera already reports that vehicle this frame
                if seen_now:  # seen by the other camera this frame
                    other = next(iter(seen_now.values()))
                    ref, gate = other["road"], self.SAME_FRAME_GATE_M
                else:
                    gap = frame_idx - veh["last_seen"]
                    ref = (veh["road"][0] + veh["vel"][0] * dt * gap, veh["road"][1] + veh["vel"][1] * dt * gap)
                    gate = self.RECENT_LOST_GATE_M if gap <= 5 else self.LONG_LOST_GATE_M
                d = math.hypot(obs["road"][0] - ref[0], obs["road"][1] - ref[1])
                if d < gate:
                    pairs.append((d, oi, key))

        used_obs, used_keys = set(), set()
        for d, oi, key in sorted(pairs):
            if oi in used_obs or key in used_keys:
                continue
            used_obs.add(oi)
            used_keys.add(key)
            obs = unbound[oi]
            self._local_to_key[(cam, obs["local_id"])] = key
            obs_by_key.setdefault(key, {})[cam] = obs

        for oi, obs in enumerate(unbound):
            if oi in used_obs:
                continue
            if math.hypot(obs["road"][0] - self.roi_center[0], obs["road"][1] - self.roi_center[1]) >= self.roi_radius:
                continue
            key = self._next_key
            self._next_key += 1
            self._vehicles[key] = {
                "id": None,
                "class_name": obs["class_name"],
                "first_seen": frame_idx,
                "last_seen": frame_idx,
                "hits": 0,
                "road": obs["road"],
                "vel": (0.0, 0.0),
                "active_s": None,
                "active_n": None,
                "conf": obs["conf"],
                "local_ids": {"south": [], "north": []},
                "trajectory_road": [],
                "history_bboxes_composite": {},
            }
            self._local_to_key[(cam, obs["local_id"])] = key
            obs_by_key.setdefault(key, {})[cam] = obs

    def _road_point(self, bx: Sequence[float], cam_pos, cam_rot, w: int, h: int) -> Optional[Tuple[float, float]]:
        """Road-plane position of a detected vehicle from its image box.

        "box75": ray through the point at 75 % of the box height intersected with a plane 0.35 m above
        the road - close to the vehicle centre from both camera directions (measured on CARLA ground
        truth: median error 0.9 m vs 1.4 m, and the South/North estimates of one car agree to 1.7 m
        vs 2.9 m). "bottom": bottom-centre of the box on the road (nearest edge of the vehicle).
        """
        u = (bx[0] + bx[2]) / 2.0
        if self.ground_point == "box75":
            v = bx[1] + 0.75 * (bx[3] - bx[1])
            return pixel_to_road_ray(u, v, cam_pos, cam_rot, fov=90.0, w=w, h=h, ground_z=0.35)
        return pixel_to_road_ray(u, float(bx[3]), cam_pos, cam_rot, fov=90.0, w=w, h=h)

    def _merge_duplicates(self, obs_by_key: Dict[int, Dict[str, Any]]) -> None:
        """Merge global vehicles that are the same car (seen by different cameras but not linked).

        Two vehicles observed this frame by *different* cameras that stay within MERGE_DIST_M for
        MERGE_FRAMES consecutive frames are merged; the confirmed / older one keeps its ID.
        """
        keys = [k for k in obs_by_key if k in self._vehicles]
        close_now = set()
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if set(obs_by_key[a]) & set(obs_by_key[b]):
                    continue  # both seen by the same camera as separate detections -> two cars
                va, vb = self._vehicles[a], self._vehicles[b]
                if math.hypot(va["road"][0] - vb["road"][0], va["road"][1] - vb["road"][1]) < self.MERGE_DIST_M:
                    pair = (min(a, b), max(a, b))
                    close_now.add(pair)
                    self._close_pairs[pair] = self._close_pairs.get(pair, 0) + 1
        self._close_pairs = {p: n for p, n in self._close_pairs.items() if p in close_now}
        for (a, b), n in list(self._close_pairs.items()):
            if n < self.MERGE_FRAMES or a not in self._vehicles or b not in self._vehicles:
                continue
            va, vb = self._vehicles[a], self._vehicles[b]
            rank = lambda v: (v["id"] is None, v["id"] if v["id"] is not None else 0, v["first_seen"])
            keep, drop = (a, b) if rank(va) <= rank(vb) else (b, a)
            vk, vd = self._vehicles[keep], self._vehicles[drop]
            for cam in ("south", "north"):
                vk["local_ids"][cam] = sorted(set(vk["local_ids"][cam]) | set(vd["local_ids"][cam]))
            if vk["active_s"] is None:
                vk["active_s"] = vd["active_s"]
            if vk["active_n"] is None:
                vk["active_n"] = vd["active_n"]
            for lk, kk in list(self._local_to_key.items()):
                if kk == drop:
                    self._local_to_key[lk] = keep
            vk["hits"] += vd["hits"]
            if vk["id"] is None and vk["hits"] >= self.min_hits:
                vk["id"] = self.next_id
                self.next_id += 1
            del self._vehicles[drop]
            self._close_pairs.pop((a, b), None)

    def _apply_observation(
        self, veh: Dict[str, Any], cam_obs: Dict[str, Dict[str, Any]], frame_idx: int, dt: float, pane_w: int
    ) -> None:
        obs_s, obs_n = cam_obs.get("south"), cam_obs.get("north")
        pts = [o["road"] for o in (obs_s, obs_n) if o is not None]
        road = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

        if veh["hits"] > 0:
            time_diff = max(dt, (frame_idx - veh["last_seen"]) * dt)
            vx = (road[0] - veh["road"][0]) / time_diff
            vy = (road[1] - veh["road"][1]) / time_diff
            veh["vel"] = (0.7 * veh["vel"][0] + 0.3 * vx, 0.7 * veh["vel"][1] + 0.3 * vy)

        best = max((o for o in (obs_s, obs_n) if o is not None), key=lambda o: o["conf"])
        veh["road"] = road
        veh["last_seen"] = frame_idx
        veh["hits"] += 1
        veh["conf"] = best["conf"]
        veh["class_name"] = best["class_name"]
        veh["active_s"] = obs_s["bbox"] if obs_s else None
        veh["active_n"] = obs_n["bbox"] if obs_n else None
        for o in (obs_s, obs_n):
            if o is not None and o["local_id"] not in veh["local_ids"][o["cam"]]:
                veh["local_ids"][o["cam"]].append(o["local_id"])
        veh["trajectory_road"].append((road[0], road[1], frame_idx * dt))
        if obs_s is not None:
            veh["history_bboxes_composite"][frame_idx] = obs_s["bbox"]
        else:
            bx = obs_n["bbox"]
            veh["history_bboxes_composite"][frame_idx] = (bx[0] + pane_w, bx[1], bx[2] + pane_w, bx[3])

        if veh["id"] is None and veh["hits"] >= self.min_hits:
            veh["id"] = self.next_id
            self.next_id += 1

    def _expire(self, frame_idx: int) -> None:
        expired = []
        for key, veh in self._vehicles.items():
            gap = frame_idx - veh["last_seen"]
            if veh["id"] is None and gap > 5:
                expired.append(key)  # unconfirmed vehicle that vanished
            elif gap > self.max_age:
                self.completed_tracks[str(veh["id"])] = veh
                expired.append(key)
        for key in expired:
            del self._vehicles[key]
        if expired:
            gone = set(expired)
            self._local_to_key = {k: v for k, v in self._local_to_key.items() if v not in gone}

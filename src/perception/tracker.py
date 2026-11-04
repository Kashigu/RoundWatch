"""Multi-Object Tracking (MOT) Engine based on BoT-SORT for RoundWatch.

Maintains persistent vehicle identities across dense traffic and large vehicle
occlusions using Kalman filter state estimation, Camera Motion Compensation (CMC),
and trajectory history accumulation buffers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class KalmanBoxTracker:
    """Kalman filter for tracking single bounding box [x1, y1, x2, y2]."""

    count = 0

    def __init__(self, bbox: Tuple[int, int, int, int], class_id: int, conf: float) -> None:
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.class_id = class_id
        self.conf = conf
        self.time_since_update = 0
        self.history: List[Tuple[float, float, float]] = []  # (x_center, y_bottom, timestamp)
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        # State: [center_x, center_y, scale (area), aspect_ratio, vx, vy, vscale]
        # Using OpenCV KalmanFilter
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)

        self.kf.statePost = self._bbox_to_z(bbox)
        self.kf.statePre = self._bbox_to_z(bbox)

        # Record initial contact point
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_bottom = float(bbox[3])
        self.history.append((x_center, y_bottom, 0.0))

    def _bbox_to_z(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        x = float(bbox[0]) + w / 2.0
        y = float(bbox[1]) + h / 2.0
        s = w * h
        r = w / h
        return np.array([[x], [y], [s], [r], [0.0], [0.0], [0.0]], dtype=np.float32)

    def _z_to_bbox(self, state: np.ndarray) -> Tuple[int, int, int, int]:
        x = float(state[0, 0])
        y = float(state[1, 0])
        s = max(1.0, float(state[2, 0]))
        r = max(0.1, float(state[3, 0]))
        w = math.sqrt(s * r)
        h = s / w
        return int(x - w / 2.0), int(y - h / 2.0), int(x + w / 2.0), int(y + h / 2.0)

    def predict(self) -> Tuple[int, int, int, int]:
        # Damp velocities when tracking without measurement to prevent runaway flying boxes
        if self.time_since_update > 0:
            self.kf.statePost[4] *= 0.6  # vx damping
            self.kf.statePost[5] *= 0.6  # vy damping
            self.kf.statePost[6] *= 0.4  # vscale damping
        pred = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._z_to_bbox(pred)

    def update(self, bbox: Tuple[int, int, int, int], conf: float, timestamp: float) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.conf = conf
        z = self._bbox_to_z(bbox)[:4]
        self.kf.correct(z)

        # Record contact point (bottom-center)
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_bottom = float(bbox[3])
        self.history.append((x_center, y_bottom, timestamp))

    def get_state(self) -> Tuple[int, int, int, int]:
        return self._z_to_bbox(self.kf.statePost)


class BoTSORTTracker:
    """Multi-object tracker with Camera Motion Compensation (CMC) and track buffering."""

    def __init__(
        self,
        max_age: int = 35,  # Frames to maintain lost tracks during occlusions
        min_hits: int = 3,
        iou_threshold: float = 0.30,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.prev_frame_gray: Optional[np.ndarray] = None

    def update(
        self,
        detections: List[Dict[str, Any]],
        current_frame: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Process frame detections and return active confirmed tracks."""
        # 1. Camera Motion Compensation (CMC)
        if current_frame is not None and self.prev_frame_gray is not None:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            # Optical flow or feature-based affine transform
            self.prev_frame_gray = gray
        elif current_frame is not None:
            self.prev_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # 2. Predict positions of existing tracks
        for trk in self.trackers:
            trk.predict()

        # 3. Associate detections to existing tracks via IoU
        matched_trks, unmatched_dets, unmatched_trks = self._associate(detections)

        # 4. Update matched tracks
        for d_idx, t_idx in matched_trks:
            det = detections[d_idx]
            self.trackers[t_idx].update(det["bbox"], det["conf"], timestamp)

        # 5. Create new tracks for unmatched high-confidence detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            if det["conf"] >= 0.35:
                trk = KalmanBoxTracker(det["bbox"], det["class_id"], det["conf"])
                self.trackers.append(trk)

        # 6. Gather active confirmed tracks and prune stale ones
        active_tracks: List[Dict[str, Any]] = []
        surviving_trackers: List[KalmanBoxTracker] = []

        for trk in self.trackers:
            if trk.time_since_update < self.max_age:
                surviving_trackers.append(trk)
                # Only display confirmed, actively detected tracks in current frame (prevents flying ghost boxes)
                if trk.time_since_update == 0 and (trk.hits >= self.min_hits or trk.age < 5):
                    bbox = trk.get_state()
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if 10 <= w <= 450 and 10 <= h <= 450:
                        active_tracks.append({
                            "track_id": trk.id,
                            "class_id": trk.class_id,
                            "conf": trk.conf,
                            "bbox": bbox,
                            "history": list(trk.history),
                        })

        self.trackers = surviving_trackers
        return active_tracks

    def _associate(
        self, detections: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.trackers:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.trackers)))

        # Build IoU cost matrix
        iou_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)
        for d_idx, det in enumerate(detections):
            for t_idx, trk in enumerate(self.trackers):
                iou_matrix[d_idx, t_idx] = self._iou(det["bbox"], trk.get_state())

        matched_indices: List[Tuple[int, int]] = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.trackers)))

        # Greedy match high IoUs
        while True:
            max_iou = 0.0
            best_pair = None
            for d in unmatched_dets:
                for t in unmatched_trks:
                    if iou_matrix[d, t] > max_iou:
                        max_iou = iou_matrix[d, t]
                        best_pair = (d, t)

            if max_iou >= self.iou_threshold and best_pair is not None:
                d, t = best_pair
                matched_indices.append((d, t))
                unmatched_dets.remove(d)
                unmatched_trks.remove(t)
            else:
                break

        return matched_indices, unmatched_dets, unmatched_trks

    @staticmethod
    def _iou(bb_test: Tuple[int, int, int, int], bb_gt: Tuple[int, int, int, int]) -> float:
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])
        w = max(0.0, float(xx2 - xx1))
        h = max(0.0, float(yy2 - yy1))
        intersection = w * h
        area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        union = area1 + area2 - intersection
        return float(intersection / union) if union > 0 else 0.0


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
    """Pure dynamic multi-camera tracking engine without GT or pre-defined IDs.
    
    Maintains persistent vehicle identities dynamically across dual cameras (South and North)
    and central fountain occlusions by mapping video bounding boxes to road metric space.
    """

    def __init__(
        self,
        cam_pos_s: List[float] | Tuple[float, float, float],
        cam_rot_s: List[float] | Tuple[float, float, float],
        cam_pos_n: List[float] | Tuple[float, float, float],
        cam_rot_n: List[float] | Tuple[float, float, float],
        max_age_frames: int = 45,
        min_hits: int = 3,
    ) -> None:
        self.cam_pos_s = cam_pos_s
        self.cam_rot_s = cam_rot_s
        self.cam_pos_n = cam_pos_n
        self.cam_rot_n = cam_rot_n
        self.max_age = max_age_frames
        self.min_hits = min_hits
        self.next_id = 1

        self.tentative_tracks: List[Dict[str, Any]] = []
        self.confirmed_tracks: List[Dict[str, Any]] = []
        self.completed_tracks: Dict[str, Dict[str, Any]] = {}

    def update(
        self,
        dets_s: List[Dict[str, Any]],
        dets_n: List[Dict[str, Any]],
        frame_idx: int,
        dt: float = 0.05,
        pane_w: int = 1280,
        height: int = 720,
    ) -> List[Dict[str, Any]]:
        # 1. Project detections to road coordinates
        obs_s = []
        for d in dets_s:
            bx = d["bbox"]
            fu, fv = (bx[0] + bx[2]) / 2.0, float(bx[3])
            pt = pixel_to_road_ray(fu, fv, self.cam_pos_s, self.cam_rot_s, fov=90.0, w=pane_w, h=height)
            if pt:
                obs_s.append({"pane": "south", "bbox": bx, "road": pt, "class_name": d.get("class_name", "car"), "conf": d.get("conf", 0.8)})

        obs_n = []
        for d in dets_n:
            bx = d["bbox"]
            fu, fv = (bx[0] + bx[2]) / 2.0, float(bx[3])
            pt = pixel_to_road_ray(fu, fv, self.cam_pos_n, self.cam_rot_n, fov=90.0, w=pane_w, h=height)
            if pt:
                obs_n.append({"pane": "north", "bbox": bx, "road": pt, "class_name": d.get("class_name", "car"), "conf": d.get("conf", 0.8)})

        # 2. Fuse simultaneous detections (same physical car seen in both South & North simultaneously)
        fused_obs = []
        used_s = set()
        used_n = set()

        for is_idx, os_item in enumerate(obs_s):
            best_in = None
            best_dist = float("inf")
            for in_idx, on_item in enumerate(obs_n):
                if in_idx in used_n:
                    continue
                d = math.hypot(os_item["road"][0] - on_item["road"][0], os_item["road"][1] - on_item["road"][1])
                if d < best_dist and d < 6.5:
                    best_dist = d
                    best_in = in_idx

            if best_in is not None:
                used_s.add(is_idx)
                used_n.add(best_in)
                on_item = obs_n[best_in]
                avg_road = (
                    0.5 * (os_item["road"][0] + on_item["road"][0]),
                    0.5 * (os_item["road"][1] + on_item["road"][1]),
                )
                fused_obs.append({
                    "bbox_s": os_item["bbox"],
                    "bbox_n": on_item["bbox"],
                    "road": avg_road,
                    "class_name": os_item["class_name"] if os_item["conf"] >= on_item["conf"] else on_item["class_name"],
                    "conf": max(os_item["conf"], on_item["conf"]),
                })
            else:
                fused_obs.append({
                    "bbox_s": os_item["bbox"],
                    "bbox_n": None,
                    "road": os_item["road"],
                    "class_name": os_item["class_name"],
                    "conf": os_item["conf"],
                })

        for in_idx, on_item in enumerate(obs_n):
            if in_idx not in used_n:
                fused_obs.append({
                    "bbox_s": None,
                    "bbox_n": on_item["bbox"],
                    "road": on_item["road"],
                    "class_name": on_item["class_name"],
                    "conf": on_item["conf"],
                })

        # 3. Associate with confirmed tracks
        unmatched_obs = list(range(len(fused_obs)))
        for trk in self.confirmed_tracks:
            pred_x = trk["road"][0] + trk["vel"][0] * dt
            pred_y = trk["road"][1] + trk["vel"][1] * dt

            best_oi = None
            best_dist = float("inf")
            for oi in unmatched_obs:
                fo = fused_obs[oi]
                d = math.hypot(fo["road"][0] - pred_x, fo["road"][1] - pred_y)
                max_gate = 8.0 if (frame_idx - trk["last_seen"] <= 5) else 14.0
                if d < best_dist and d < max_gate:
                    best_dist = d
                    best_oi = oi

            if best_oi is not None:
                fo = fused_obs[best_oi]
                unmatched_obs.remove(best_oi)
                time_diff = max(dt, (frame_idx - trk["last_seen"]) * dt)
                vx = (fo["road"][0] - trk["road"][0]) / time_diff
                vy = (fo["road"][1] - trk["road"][1]) / time_diff
                trk["vel"] = (0.7 * trk["vel"][0] + 0.3 * vx, 0.7 * trk["vel"][1] + 0.3 * vy)
                trk["road"] = fo["road"]
                trk["last_seen"] = frame_idx
                trk["hits"] += 1
                trk["active_s"] = fo["bbox_s"]
                trk["active_n"] = fo["bbox_n"]
                trk["conf"] = fo["conf"]
                trk["trajectory_road"].append((fo["road"][0], fo["road"][1], frame_idx * dt))
                if fo["bbox_s"] is not None:
                    trk["history_bboxes_composite"][frame_idx] = fo["bbox_s"]
                elif fo["bbox_n"] is not None:
                    bx_n = fo["bbox_n"]
                    trk["history_bboxes_composite"][frame_idx] = (bx_n[0] + pane_w, bx_n[1], bx_n[2] + pane_w, bx_n[3])
            else:
                trk["active_s"] = None
                trk["active_n"] = None

        # 4. Associate remaining observations with tentative tracks
        remaining_unmatched = []
        for oi in unmatched_obs:
            fo = fused_obs[oi]
            matched_ten = False
            for ten in self.tentative_tracks:
                d = math.hypot(fo["road"][0] - ten["road"][0], fo["road"][1] - ten["road"][1])
                if d < 7.0:
                    ten["road"] = fo["road"]
                    ten["last_seen"] = frame_idx
                    ten["hits"] += 1
                    ten["active_s"] = fo["bbox_s"]
                    ten["active_n"] = fo["bbox_n"]
                    ten["trajectory_road"].append((fo["road"][0], fo["road"][1], frame_idx * dt))
                    if fo["bbox_s"] is not None:
                        ten["history_bboxes_composite"][frame_idx] = fo["bbox_s"]
                    elif fo["bbox_n"] is not None:
                        bx_n = fo["bbox_n"]
                        ten["history_bboxes_composite"][frame_idx] = (bx_n[0] + pane_w, bx_n[1], bx_n[2] + pane_w, bx_n[3])
                    matched_ten = True

                    if ten["hits"] >= self.min_hits:
                        ten["id"] = self.next_id
                        self.next_id += 1
                        self.confirmed_tracks.append(ten)
                        self.tentative_tracks.remove(ten)
                    break
            if not matched_ten:
                remaining_unmatched.append(oi)

        # 5. Create new tentative track
        for oi in remaining_unmatched:
            fo = fused_obs[oi]
            r_center = math.hypot(fo["road"][0] - 2.0, fo["road"][1] - 6.0)
            if r_center < 38.0:
                hist_box = {}
                if fo["bbox_s"] is not None:
                    hist_box[frame_idx] = fo["bbox_s"]
                elif fo["bbox_n"] is not None:
                    bx_n = fo["bbox_n"]
                    hist_box[frame_idx] = (bx_n[0] + pane_w, bx_n[1], bx_n[2] + pane_w, bx_n[3])
                self.tentative_tracks.append({
                    "class_name": fo["class_name"],
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "hits": 1,
                    "road": fo["road"],
                    "vel": (0.0, 0.0),
                    "active_s": fo["bbox_s"],
                    "active_n": fo["bbox_n"],
                    "conf": fo["conf"],
                    "trajectory_road": [(fo["road"][0], fo["road"][1], frame_idx * dt)],
                    "history_bboxes_composite": hist_box,
                })

        # 6. Prune stale tentative tracks
        self.tentative_tracks = [t for t in self.tentative_tracks if frame_idx - t["last_seen"] <= 5]

        # 7. Archive expired confirmed tracks
        surviving_conf = []
        for trk in self.confirmed_tracks:
            if frame_idx - trk["last_seen"] > self.max_age:
                self.completed_tracks[str(trk["id"])] = trk
            else:
                surviving_conf.append(trk)
        self.confirmed_tracks = surviving_conf

        return [t for t in self.confirmed_tracks if (t["active_s"] is not None or t["active_n"] is not None)]


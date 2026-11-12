"""Evaluate detection + per-camera BoT-SORT + cross-camera fusion against recorded ground truth.

Ground truth comes from scripts/record_tracking_dataset.py (visible 2-D boxes from CARLA
instance segmentation + true world positions). Reports, per camera: detection recall/precision,
local ID switches and why vehicles were missed; globally (after fusion): MOTA, ID switches,
coverage of observable vehicles and "lost in both cameras" episodes with their cause.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CAMS = ("south", "north")


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def assign(cost: np.ndarray, limit: float) -> List[Tuple[int, int]]:
    """Min-cost assignment keeping only pairs with cost < limit."""
    if cost.size == 0:
        return []
    import lap

    _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=limit)
    return [(i, j) for i, j in enumerate(x) if j >= 0 and cost[i, j] < limit]


@dataclass
class EvalConfig:
    model: str = "yolov8s.pt"
    res: Optional[Tuple[int, int]] = None      # resize frames to (w, h); None = recorded resolution
    imgsz: Optional[int] = None                # YOLO inference size; None = detector default
    conf: float = 0.25
    tracker_cfg: str = "configs/botsort.yaml"
    fusion: Dict[str, Any] = field(default_factory=lambda: {"max_age_frames": 40, "min_hits": 2})
    min_vis_px_1080: int = 250                 # a GT vehicle counts as visible from this many pixels at 1080p
    det_iou: float = 0.3
    world_gate_m: float = 5.0
    max_frames: Optional[int] = None


def _bucket(v: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for e, l in zip(edges, labels):
        if v < e:
            return l
    return labels[-1]


def evaluate(data_dir: str, cfg: EvalConfig, verbose: bool = True) -> Dict[str, Any]:
    from src.perception.detector import VehicleDetector
    from src.perception.tracker import DynamicMultiCameraTracker

    data = Path(data_dir)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    W0, H0 = meta["width"], meta["height"]
    W, H = cfg.res or (W0, H0)
    sx, sy = W / W0, H / H0
    min_px = cfg.min_vis_px_1080 * (W * H) / (1920 * 1080)
    dt = 1.0 / meta["fps"]
    cams = meta["cameras"]
    cam_xy = {c: cams[c]["position"][:2] for c in CAMS}

    dets = {c: VehicleDetector(model_name=cfg.model, confidence_thresh=cfg.conf, device="cuda:0", imgsz=cfg.imgsz)
            for c in CAMS}
    from src.simulation.maps import get_map

    roi_center, roi_radius = get_map(meta["map"]).tracking_roi
    fusion_kw = {"roi_center": roi_center, "roi_radius": roi_radius, **cfg.fusion}
    fusion = DynamicMultiCameraTracker(cams["south"]["position"], cams["south"]["rotation"],
                                       cams["north"]["position"], cams["north"]["rotation"], **fusion_kw)

    gt_lines = (data / "gt.jsonl").read_text(encoding="utf-8").splitlines()
    if cfg.max_frames:
        gt_lines = gt_lines[: cfg.max_frames]

    cam_stats = {c: Counter() for c in CAMS}
    miss_reasons = {c: Counter() for c in CAMS}
    last_local = {c: {} for c in CAMS}
    g = Counter()
    last_global: Dict[str, int] = {}
    unmatched_run: Dict[str, int] = defaultdict(int)
    lost_episodes: List[Dict[str, Any]] = []
    idsw_kinds = Counter()
    gid_to_gt_seen: Dict[int, set] = defaultdict(set)
    t_infer = 0.0

    for li, line in enumerate(gt_lines):
        fr = json.loads(line)
        f = fr["frame"]
        imgs = {}
        for c in CAMS:
            im = cv2.imread(str(data / c / f"{f:06d}.jpg"))
            if (W, H) != (W0, H0):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            imgs[c] = im
        t0 = time.time()
        tracks = {c: dets[c].track(imgs[c], tracker_cfg=cfg.tracker_cfg) for c in CAMS}
        active = fusion.update(tracks["south"], tracks["north"], frame_idx=f, dt=dt, pane_w=W, height=H)
        t_infer += time.time() - t0

        detected_any: Dict[str, bool] = defaultdict(bool)
        visible_any: Dict[str, bool] = defaultdict(bool)
        for c in CAMS:
            gts = []
            for v in fr["vehicles"]:
                o = v.get(c)
                if o and o["vis_box"] and o["vis_px"] * sx * sy >= min_px:
                    b = [o["vis_box"][0] * sx, o["vis_box"][1] * sy, o["vis_box"][2] * sx, o["vis_box"][3] * sy]
                    full = o["full_box"]
                    full_area = max(1.0, (full[2] - full[0]) * (full[3] - full[1])) if full else o["vis_px"]
                    gts.append((v, b, o["vis_px"] / full_area))
            for v, _b, _r in gts:
                visible_any[v["sumo_id"]] = True
            tr = tracks[c]
            cost = np.array([[1.0 - iou(b, t["bbox"]) for t in tr] for _v, b, _r in gts]) if gts and tr else np.zeros((len(gts), len(tr)))
            pairs = assign(cost, 1.0 - cfg.det_iou) if gts and tr else []
            matched_g = {i for i, _ in pairs}
            st = cam_stats[c]
            st["gt"] += len(gts)
            st["tp"] += len(pairs)
            st["fp"] += len(tr) - len(pairs)
            owner_now = {tr[j]["track_id"]: gts[i][0]["sumo_id"] for i, j in pairs}
            for i, j in pairs:
                sid = gts[i][0]["sumo_id"]
                detected_any[sid] = True
                lid = tr[j]["track_id"]
                if sid in last_local[c] and last_local[c][sid] != lid:
                    st["idsw"] += 1
                    old = last_local[c][sid]
                    # swap: the old id now follows another vehicle, or this new id belonged to another one before
                    took_other = any(last_local[c].get(o) == lid for o in last_local[c] if o != sid)
                    st["idsw_swap" if (owner_now.get(old, sid) != sid or took_other) else "idsw_fragment"] += 1
                last_local[c][sid] = lid
            for i, (v, b, ratio) in enumerate(gts):
                if i in matched_g:
                    continue
                st["fn"] += 1
                d = math.hypot(v["x"] - cam_xy[c][0], v["y"] - cam_xy[c][1])
                h = b[3] - b[1]
                miss_reasons[c][f"occlusion:{_bucket(ratio, (0.4, 0.75), ('heavy', 'partial', 'none'))}"] += 1
                miss_reasons[c][f"distance:{_bucket(d, (25, 45, 65), ('<25m', '25-45m', '45-65m', '>65m'))}"] += 1
                miss_reasons[c][f"height_px@{H}:{_bucket(h, (20, 40, 80), ('<20', '20-40', '40-80', '>80'))}"] += 1

        # global evaluation (fused tracks vs GT world positions), only vehicles visible in >= 1 camera
        obs = [v for v in fr["vehicles"] if visible_any[v["sumo_id"]]
               and math.hypot(v["x"] - roi_center[0], v["y"] - roi_center[1]) < roi_radius]
        cost = np.array([[math.hypot(v["x"] - t["road"][0], v["y"] - t["road"][1]) for t in active] for v in obs]) \
            if obs and active else np.zeros((len(obs), len(active)))
        pairs = assign(cost, cfg.world_gate_m) if obs and active else []
        g["gt"] += len(obs)
        g["tp"] += len(pairs)
        g["fp"] += len(active) - len(pairs)
        matched_tracks = {j for _i, j in pairs}
        for j, t in enumerate(active):
            if j in matched_tracks:
                continue
            near = min((math.hypot(v["x"] - t["road"][0], v["y"] - t["road"][1]) for v in fr["vehicles"]), default=1e9)
            g["fp_duplicate_or_offset" if near < 2 * cfg.world_gate_m else "fp_ghost"] += 1
        matched_obs = {}
        for i, j in pairs:
            sid, gid = obs[i]["sumo_id"], active[j]["id"]
            matched_obs[sid] = gid
            gid_to_gt_seen[gid].add(sid)
            if sid in last_global and last_global[sid] != gid:
                g["idsw"] += 1
                idsw_kinds["after_gap" if unmatched_run[sid] > 0 else "direct"] += 1
            last_global[sid] = gid
        for v in obs:
            sid = v["sumo_id"]
            if sid in matched_obs:
                if unmatched_run[sid] >= 3:
                    lost_episodes.append({"sumo_id": sid, "frames": unmatched_run[sid], **unmatched_run.get(("why", sid), {})})
                unmatched_run[sid] = 0
                unmatched_run.pop(("why", sid), None)
            else:
                g["fn"] += 1
                unmatched_run[sid] += 1
                why = unmatched_run.setdefault(("why", sid), Counter())
                why["detected_in_a_camera" if detected_any[sid] else "not_detected_any_camera"] += 1
        if verbose and (li + 1) % 400 == 0:
            print(f"  [eval] {li + 1}/{len(gt_lines)} frames")

    for sid, run in list(unmatched_run.items()):  # episodes still open at the end of the recording
        if isinstance(sid, str) and run >= 3:
            lost_episodes.append({"sumo_id": sid, "frames": run, **unmatched_run.get(("why", sid), {})})
    n = len(gt_lines)
    def cam_summary(st):
        return {"recall": round(st["tp"] / max(1, st["gt"]), 3), "precision": round(st["tp"] / max(1, st["tp"] + st["fp"]), 3),
                "id_switches": st["idsw"], "idsw_swap_between_vehicles": st["idsw_swap"], "idsw_fragmentation": st["idsw_fragment"], "gt_boxes": st["gt"], "missed": st["fn"], "false_pos": st["fp"]}
    lost_cause = Counter()
    for ep in lost_episodes:
        for k, v in ep.items():
            if k not in ("sumo_id", "frames"):
                lost_cause[k] += v
    result = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "frames": n,
        "fps_pipeline": round(n / t_infer, 1) if t_infer else None,
        "per_camera": {c: cam_summary(cam_stats[c]) for c in CAMS},
        "miss_reasons": {c: dict(sorted(miss_reasons[c].items())) for c in CAMS},
        "global": {
            "MOTA": round(1.0 - (g["fn"] + g["fp"] + g["idsw"]) / max(1, g["gt"]), 3),
            "coverage": round(g["tp"] / max(1, g["gt"]), 3),
            "id_switches": g["idsw"], "id_switch_kinds": dict(idsw_kinds),
            "false_tracks": g["fp"], "false_tracks_near_a_vehicle": g["fp_duplicate_or_offset"], "false_tracks_ghost": g["fp_ghost"], "gt_vehicle_frames": g["gt"],
            "vehicles": len(last_global),
            "global_ids_used": len(gid_to_gt_seen),
            "ids_shared_by_2plus_vehicles": sum(1 for s in gid_to_gt_seen.values() if len(s) > 1),
            "lost_episodes_3plus_frames": len(lost_episodes),
            "lost_frames_by_cause": dict(lost_cause),
        },
    }
    return result

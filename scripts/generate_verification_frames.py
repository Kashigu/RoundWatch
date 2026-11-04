import os
import sys
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.compliance.geometry import RoundaboutGeometry
from src.perception.homography import RoadwayHomography
from src.visualization.ar_overlay import AROverlayEngine

geom = RoundaboutGeometry()
homography = RoadwayHomography()
with open("configs/homography_town03.json", "r") as f:
    cfg = json.load(f)
homography.H = np.array(cfg["H_matrix"], dtype=np.float64)
homography.H_inv = np.array(cfg["H_inv_matrix"], dtype=np.float64)

ar_engine = AROverlayEngine(geometry=geom, homography=homography, corridor_width_m=3.6, alpha=0.45)

compliance_data = {}
with open("output/compliance_results.json", "r", encoding="utf-8") as f:
    raw_comp = json.load(f)
    for k, v in raw_comp.items():
        compliance_data[int(k)] = v

tracks_data = {}
with open("output/tracks.json", "r", encoding="utf-8") as f:
    raw_trk = json.load(f)
    for k, v in raw_trk.items():
        tracks_data[int(k)] = v

cap = cv2.VideoCapture("data/recordings/carla_sumo_roundabout.mp4")

def render_frame_at(frame_idx, sel_id=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None, []
    active_tracks = []
    for tid, tdata in tracks_data.items():
        ff = tdata.get("first_frame", 0)
        lf = tdata.get("last_frame", 0)
        if ff <= frame_idx <= lf:
            offset = frame_idx - ff
            bboxes = tdata.get("history_bboxes", [])
            hist_px = tdata.get("history_pixels", [])
            cur_bbox = bboxes[offset] if offset < len(bboxes) else (0, 0, 0, 0)
            active_tracks.append({
                "track_id": tid,
                "bbox": cur_bbox,
                "history_pixels": hist_px[: offset + 1],
            })
    rendered = ar_engine.render_overlay(
        frame=frame,
        active_tracks=active_tracks,
        compliance_results=compliance_data,
        selected_track_id=sel_id,
        frame_idx=frame_idx,
        timestamp=frame_idx / 20.0
    )
    return rendered, [t["track_id"] for t in active_tracks]

# Frame 0: Clean startup
f0, _ = render_frame_at(0, None)
if f0 is not None:
    cv2.imwrite("fixed_frame_0.png", f0)
    print("Saved fixed_frame_0.png")

# Frame 250: Unclicked & Clicked
f250_unsel, act250 = render_frame_at(250, None)
if f250_unsel is not None:
    cv2.imwrite("fixed_frame_250_unselected.png", f250_unsel)
    print("Saved fixed_frame_250_unselected.png")
    if act250:
        f250_sel, _ = render_frame_at(250, act250[0])
        cv2.imwrite("fixed_frame_250_selected.png", f250_sel)
        print(f"Saved fixed_frame_250_selected.png (selected {act250[0]})")

# Frame 450: Western quadrant transit
f450_unsel, act450 = render_frame_at(450, None)
if f450_unsel is not None:
    cv2.imwrite("fixed_frame_450_transit.png", f450_unsel)
    print("Saved fixed_frame_450_transit.png")
    if act450:
        f450_sel, _ = render_frame_at(450, act450[0])
        cv2.imwrite("fixed_frame_450_selected.png", f450_sel)
        print(f"Saved fixed_frame_450_selected.png (selected {act450[0]})")

cap.release()

# Copy to artifacts directory
artifact_dir = r"C:\Users\orlan\.gemini\antigravity\brain\dc5e9231-1ac9-4f55-b832-205cbb1301e5"
for name in ["fixed_frame_0.png", "fixed_frame_250_unselected.png", "fixed_frame_250_selected.png", "fixed_frame_450_transit.png", "fixed_frame_450_selected.png"]:
    if os.path.exists(name):
        shutil.copy(name, os.path.join(artifact_dir, name))
        print(f"Copied {name} to artifact directory")

import cv2
import json
import numpy as np
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath("."))

from runtime_paths import artifact_directory

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

# Pick key frames to illustrate the full simulation journey
frames_to_export = [
    (100, None, "step_01_frame_100.png"),
    (200, None, "step_02_frame_200.png"),
    (300, 4, "step_03_frame_300_infraction.png"),
    (420, None, "step_04_frame_420_truck_transit.png"),
    (600, 8, "step_05_frame_600_transit.png"),
    (750, None, "step_06_frame_750_exit.png"),
]

artifact_dir = str(artifact_directory())

for f_idx, sel_id, filename in frames_to_export:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if ret:
        active_tracks = []
        for tid, tdata in tracks_data.items():
            ff = tdata.get("first_frame", 0)
            lf = tdata.get("last_frame", 0)
            if ff <= f_idx <= lf:
                offset = f_idx - ff
                bboxes = tdata.get("history_bboxes", [])
                hist_px = tdata.get("history_pixels", [])
                cur_bbox = bboxes[offset] if offset < len(bboxes) else (0, 0, 0, 0)
                active_tracks.append({
                    "track_id": tid,
                    "bbox": cur_bbox,
                    "history_pixels": hist_px[: offset + 1],
                })
        # If sel_id is specified but not active, pick first active
        final_sel = sel_id if any(t["track_id"] == sel_id for t in active_tracks) else (active_tracks[0]["track_id"] if (sel_id is not None and active_tracks) else None)
        out = ar_engine.render_overlay(
            frame=frame,
            active_tracks=active_tracks,
            compliance_results=compliance_data,
            selected_track_id=final_sel,
            frame_idx=f_idx,
            timestamp=f_idx / 20.0
        )
        cv2.imwrite(filename, out)
        shutil.copy(filename, os.path.join(artifact_dir, filename))
        print(f"Exported {filename} (sel={final_sel})")

cap.release()
print("All key frames exported!")

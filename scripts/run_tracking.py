"""Vehicle Detection and Multi-Object Tracking (BoT-SORT + YOLO) Pipeline Runner.

Ingests video stream (CARLA or Mock), detects vehicles with YOLO on CUDA,
tracks trajectories with BoT-SORT, maps foot contact points to road plane
coordinates via calibrated homography, and exports output/tracks.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.perception.detector import VehicleDetector
from src.perception.homography import RoadwayHomography


def run_tracking_pipeline(
    video_path: str = "data/recordings/carla_feed_h8m.mp4",
    homography_config: str = "configs/homography_town03.json",
    output_dir: str = "output",
    render_video: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end detection and tracking on video stream."""
    video_file = Path(video_path)
    if not video_file.exists():
        # Fallback to mock video if recording not found
        fallback = Path("data/mock/synthetic_roundabout_feed.mp4")
        if fallback.exists():
            print(f"[Tracker] {video_file} not found, falling back to {fallback}")
            video_file = fallback
        else:
            raise FileNotFoundError(f"Neither {video_file} nor {fallback} exists.")

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_file}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Tracker] Ingesting {video_file.name}: {total_frames} frames @ {fps:.1f} FPS, {width}x{height}")

    # Load Homography
    homography = RoadwayHomography()
    if Path(homography_config).exists():
        with open(homography_config, "r") as f:
            cfg = json.load(f)
        homography.H = np.array(cfg["H_matrix"], dtype=np.float64)
        homography.H_inv = np.array(cfg["H_inv_matrix"], dtype=np.float64)
        print("[Tracker] Loaded calibrated homography matrix.")

    # Initialize Perception Modules
    # YOLO + official BoT-SORT (GMC + native ReID), config in configs/botsort.yaml
    detector = VehicleDetector(model_name="yolov8n.pt", confidence_thresh=0.30, device="cuda:0")

    writer = None
    output_video_path = Path(output_dir) / "tracked_roundabout_feed.mp4"
    if render_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Trajectory storage: track_id -> list of records
    accumulated_tracks: Dict[int, Dict[str, Any]] = {}
    frame_idx = 0
    start_time = time.time()

    colors = [
        (255, 105, 65), (50, 205, 50), (255, 191, 0), (238, 130, 238),
        (0, 191, 255), (255, 140, 0), (147, 112, 219), (0, 255, 127)
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # 1-2. YOLO Vehicle Detection + BoT-SORT Multi-Object Tracking
        active_tracks = detector.track(frame)

        # 3. Trajectory Mapping and Visual Annotation
        annotated_frame = frame.copy() if render_video else None

        for trk in active_tracks:
            tid = trk["track_id"]
            bbox = trk["bbox"]
            cls_name = detector.VEHICLE_CLASSES.get(trk["class_id"], "vehicle")
            color = colors[tid % len(colors)]

            # Map foot contact point to road plane coordinates (X, Y)
            foot_u = (bbox[0] + bbox[2]) / 2.0
            foot_v = float(bbox[3])
            road_x, road_y = homography.pixel_to_road(foot_u, foot_v)

            if tid not in accumulated_tracks:
                accumulated_tracks[tid] = {
                    "track_id": tid,
                    "class_name": cls_name,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "history_pixels": [],
                    "trajectory_road": [],
                }

            rec = accumulated_tracks[tid]
            rec["last_frame"] = frame_idx
            rec["history_pixels"].append((int(round(foot_u)), int(round(foot_v))))
            rec["trajectory_road"].append((round(road_x, 3), round(road_y, 3), round(timestamp, 3)))

            # Render bounding box and track ID
            if annotated_frame is not None:
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                tag = f"ID:{tid} {cls_name} ({trk['conf']:.2f})"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(
                    annotated_frame,
                    (bbox[0], max(0, bbox[1] - th - 6)),
                    (bbox[0] + tw + 6, max(th + 6, bbox[1])),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    tag,
                    (bbox[0] + 3, max(th + 2, bbox[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                # Draw trajectory trail
                trail = rec["history_pixels"]
                for k in range(1, len(trail)):
                    cv2.line(annotated_frame, trail[k - 1], trail[k], color, 2)

        if writer is not None and annotated_frame is not None:
            # HUD overlay
            hud_text = f"RoundWatch Tracking | Frame: {frame_idx}/{total_frames} | Active Tracks: {len(active_tracks)}"
            cv2.putText(annotated_frame, hud_text, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            writer.write(annotated_frame)

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - start_time
            cur_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_idx:3d}/{total_frames} | Active: {len(active_tracks):2d} | Speed: {cur_fps:.1f} FPS")

    cap.release()
    if writer is not None:
        writer.release()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("TRACKING PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Unique Vehicles Tracked: {len(accumulated_tracks)}")
    print(f"Processing Time:        {total_time:.2f} s ({avg_fps:.1f} FPS)")
    if render_video:
        print(f"Annotated Video Export: {output_video_path}")

    # Export tracks JSON
    tracks_json_path = Path(output_dir) / "tracks.json"
    with open(tracks_json_path, "w", encoding="utf-8") as f:
        json.dump(accumulated_tracks, f, indent=2)
    print(f"Exported Track Metadata: {tracks_json_path}")
    print("=" * 60)

    return {
        "total_frames": frame_idx,
        "num_tracks": len(accumulated_tracks),
        "fps": round(avg_fps, 1),
        "tracks_file": str(tracks_json_path),
        "video_file": str(output_video_path) if render_video else None,
    }


if __name__ == "__main__":
    run_tracking_pipeline()


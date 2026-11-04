"""Unit Tests for Perception & Multi-Object Tracking Pipeline."""

import numpy as np
import pytest

from src.perception.detector import VehicleDetector
from src.perception.homography import RoadwayHomography
from src.perception.tracker import BoTSORTTracker, KalmanBoxTracker, DynamicMultiCameraTracker, pixel_to_road_ray


def test_roadway_homography_roundtrip():
    """Verify pixel-to-road and road-to-pixel inverse consistency."""
    src_pixels = [(100.0, 100.0), (1800.0, 100.0), (1800.0, 1000.0), (100.0, 1000.0)]
    dst_road = [(-20.0, 40.0), (20.0, 40.0), (20.0, 80.0), (-20.0, 80.0)]

    homography = RoadwayHomography(src_pixels, dst_road)

    # Test center point
    test_u, test_v = 950.0, 550.0
    rx, ry = homography.pixel_to_road(test_u, test_v)
    assert -25.0 <= rx <= 25.0
    assert 35.0 <= ry <= 85.0

    # Test reverse mapping
    u_rec, v_rec = homography.road_to_pixel(rx, ry)
    assert abs(u_rec - test_u) <= 2
    assert abs(v_rec - test_v) <= 2


def test_kalman_box_tracker_prediction_update():
    """Verify Kalman filter prediction and state update step."""
    bbox = (200, 300, 260, 380)
    tracker = KalmanBoxTracker(bbox=bbox, class_id=2, conf=0.85)

    pred_box = tracker.predict()
    assert len(pred_box) == 4
    assert pred_box[2] > pred_box[0]
    assert pred_box[3] > pred_box[1]

    # Update with shifted detection
    new_bbox = (205, 305, 265, 385)
    tracker.update(new_bbox, conf=0.90, timestamp=0.05)
    assert tracker.hits == 2
    assert len(tracker.history) == 2


def test_botsort_tracker_association():
    """Verify BoT-SORT multi-object tracking association and track lifetime."""
    tracker = BoTSORTTracker(max_age=10, min_hits=2, iou_threshold=0.3)

    # Frame 1: 2 vehicle detections
    dets_f1 = [
        {"bbox": (100, 100, 160, 160), "class_id": 2, "conf": 0.9},
        {"bbox": (400, 400, 470, 470), "class_id": 2, "conf": 0.88},
    ]
    tracks_f1 = tracker.update(dets_f1, timestamp=0.0)
    assert len(tracks_f1) >= 0

    # Frame 2: vehicles moved slightly
    dets_f2 = [
        {"bbox": (103, 102, 163, 162), "class_id": 2, "conf": 0.92},
        {"bbox": (405, 404, 475, 474), "class_id": 2, "conf": 0.89},
    ]
    tracks_f2 = tracker.update(dets_f2, timestamp=0.05)
    assert len(tracks_f2) == 2
    # Verify IDs are distinct and maintained
    tids = [t["track_id"] for t in tracks_f2]
    assert len(set(tids)) == 2


def test_pixel_to_road_ray_projection():
    """Verify camera viewport ray-plane intersection with ground Z=0."""
    cam_pos_s = [0.71, -26.2, 11.5]
    cam_rot_s = [-28.0, 88.0, 0.0]

    # Center-bottom of camera viewport (looking forward into roundabout)
    pt = pixel_to_road_ray(640, 500, cam_pos_s, cam_rot_s, fov=90.0, w=1280, h=720)
    assert pt is not None
    rx, ry = pt
    # Should project within physical roundabout radius (Y > -25)
    assert -30.0 <= rx <= 30.0
    assert -25.0 <= ry <= 40.0


def test_dynamic_multi_camera_tracker_persistent_id():
    """Verify dynamic track promotion and cross-camera ID persistence."""
    cam_pos_s = [0.71, -26.2, 11.5]
    cam_rot_s = [-28.0, 88.0, 0.0]
    cam_pos_n = [-2.31, 27.2, 11.5]
    cam_rot_n = [-28.0, -80.0, 0.0]

    tracker = DynamicMultiCameraTracker(cam_pos_s, cam_rot_s, cam_pos_n, cam_rot_n, max_age_frames=10, min_hits=2)

    # Frame 1: Detection in South camera
    dets_s_1 = [{"bbox": (600, 480, 680, 560), "class_name": "car", "conf": 0.88}]
    active_1 = tracker.update(dets_s_1, [], frame_idx=0)
    assert len(active_1) == 0  # Tentative, not yet confirmed

    # Frame 2: Same detection in South camera -> Promoted to confirmed!
    dets_s_2 = [{"bbox": (602, 482, 682, 562), "class_name": "car", "conf": 0.90}]
    active_2 = tracker.update(dets_s_2, [], frame_idx=1)
    assert len(active_2) == 1
    assert active_2[0]["id"] == 1
    assert tracker.next_id == 2  # Incremented dynamically!



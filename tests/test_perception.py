"""Unit Tests for Perception & Multi-Object Tracking Pipeline."""

import numpy as np
import pytest
from types import SimpleNamespace

import src.perception.detector as detector_module
from src.perception.detector import VehicleDetector
from src.perception.homography import RoadwayHomography
from src.perception.tracker import BoTSORTTracker, DynamicMultiCameraTracker, pixel_to_road_ray


def test_vehicle_detector_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(
        detector_module,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(detector_module, "YOLO", None)

    with pytest.warns(RuntimeWarning, match="using CPU"):
        detector = VehicleDetector(device="cuda:0")

    assert detector.device == "cpu"


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


def test_botsort_tracker_association():
    """Official BoT-SORT (via adapter) keeps distinct, persistent IDs across frames."""
    tracker = BoTSORTTracker()

    def dets(offset):
        return [
            {"bbox": (100 + offset, 100 + offset, 160 + offset, 160 + offset), "class_id": 2, "conf": 0.9},
            {"bbox": (400 + offset, 400 + offset, 470 + offset, 470 + offset), "class_id": 7, "conf": 0.88},
        ]

    ids_per_frame = []
    for f in range(5):
        tracks = tracker.update(dets(2 * f))
        ids_per_frame.append({t["track_id"]: t["class_name"] for t in tracks})

    assert len(ids_per_frame[-1]) == 2
    assert ids_per_frame[-1] == ids_per_frame[-2]  # IDs stable frame to frame
    assert sorted(ids_per_frame[-1].values()) == ["car", "truck"]


def test_botsort_tracker_survives_short_occlusion():
    """A track missing for a few frames is recovered with the same ID (track_buffer)."""
    tracker = BoTSORTTracker()
    det = lambda x: [{"bbox": (x, 200, x + 60, 260), "class_id": 2, "conf": 0.9}]
    for f in range(5):
        tracks = tracker.update(det(100 + 3 * f))
    tid = tracks[0]["track_id"]
    for _ in range(4):
        assert tracker.update([]) == []
    tracks = tracker.update(det(100 + 3 * 9))
    assert len(tracks) == 1 and tracks[0]["track_id"] == tid


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


CAM_S = ([0.71, -26.2, 11.5], [-28.0, 88.0, 0.0])
CAM_N = ([-2.31, 27.2, 11.5], [-28.0, -80.0, 0.0])


def _bbox_for_road_point(cam, x, y, w=1280, h=720, half=30):
    """Find a bbox whose bottom-centre projects onto road point (x, y) for the given camera."""
    best = None
    for u in range(0, w, 4):
        for v in range(0, h, 4):
            pt = pixel_to_road_ray(u, v, cam[0], cam[1], w=w, h=h)
            if pt is None:
                continue
            d = (pt[0] - x) ** 2 + (pt[1] - y) ** 2
            if best is None or d < best[0]:
                best = (d, u, v)
    _, u, v = best
    return (u - half, v - 2 * half, u + half, v)


def test_dynamic_multi_camera_tracker_persistent_id():
    """Global IDs are confirmed after min_hits and assigned sequentially."""
    tracker = DynamicMultiCameraTracker(*CAM_S, *CAM_N, max_age_frames=10, min_hits=2)

    t1 = [{"track_id": 7, "bbox": (600, 480, 680, 560), "class_name": "car", "conf": 0.88}]
    assert tracker.update(t1, [], frame_idx=0) == []  # not yet confirmed

    t2 = [{"track_id": 7, "bbox": (602, 482, 682, 562), "class_name": "car", "conf": 0.90}]
    active = tracker.update(t2, [], frame_idx=1)
    assert len(active) == 1
    assert active[0]["id"] == 1
    assert active[0]["local_ids"]["south"] == [7]
    assert tracker.next_id == 2


def test_multi_camera_fusion_same_vehicle_both_cameras():
    """The same car seen by South and North (different BoT-SORT IDs) gets one global ID."""
    tracker = DynamicMultiCameraTracker(*CAM_S, *CAM_N, max_age_frames=10, min_hits=1)
    bs = _bbox_for_road_point(CAM_S, 0.0, 18.0)
    bn = _bbox_for_road_point(CAM_N, 0.0, 18.0)
    active = tracker.update(
        [{"track_id": 3, "bbox": bs, "class_name": "car", "conf": 0.9}],
        [{"track_id": 11, "bbox": bn, "class_name": "car", "conf": 0.8}],
        frame_idx=0,
    )
    assert len(active) == 1
    assert active[0]["active_s"] is not None and active[0]["active_n"] is not None
    assert active[0]["local_ids"] == {"south": [3], "north": [11]}


def test_multi_camera_handover_keeps_global_id():
    """When a car leaves the South view and appears in the North one, its global ID is kept."""
    tracker = DynamicMultiCameraTracker(*CAM_S, *CAM_N, max_age_frames=40, min_hits=1)
    bs = _bbox_for_road_point(CAM_S, 10.0, 15.0)
    active = tracker.update([{"track_id": 1, "bbox": bs, "class_name": "car", "conf": 0.9}], [], frame_idx=0)
    gid = active[0]["id"]

    # Occluded for 10 frames, then the North camera picks it up close to where it was lost
    for f in range(1, 11):
        tracker.update([], [], frame_idx=f)
    bn = _bbox_for_road_point(CAM_N, 8.0, 17.0)
    active = tracker.update([], [{"track_id": 5, "bbox": bn, "class_name": "car", "conf": 0.9}], frame_idx=11)
    assert len(active) == 1
    assert active[0]["id"] == gid
    assert tracker.next_id == 2

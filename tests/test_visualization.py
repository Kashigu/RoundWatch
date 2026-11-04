"""Unit Tests for AR Overlay Visualization & Evaluation Benchmark."""

import numpy as np
import pytest

from src.compliance.geometry import RoundaboutGeometry
from src.evaluation.benchmark import TrackingBenchmark
from src.perception.homography import RoadwayHomography
from src.visualization.ar_overlay import AROverlayEngine


def test_ar_overlay_corridor_generation():
    """Verify AR corridor generation and polygon projection."""
    geom = RoundaboutGeometry()
    homography = RoadwayHomography()
    engine = AROverlayEngine(geometry=geom, homography=homography)

    corridor = engine.generate_legal_corridor_road(entry_leg=0, exit_leg=1, num_pts=20)
    assert len(corridor) == 20

    poly = engine.generate_corridor_polygon(corridor)
    assert len(poly) > 0
    assert poly.shape[1] == 2


def test_tracking_benchmark_mota_calculation():
    """Verify CLEAR MOT metrics calculation with known ground-truth and predictions."""
    evaluator = TrackingBenchmark()

    # Exact matches -> MOTA = 1.0
    gt = [
        {"frame_idx": 0, "vehicle_id": 1, "bbox": [100, 100, 150, 150]},
        {"frame_idx": 1, "vehicle_id": 1, "bbox": [102, 102, 152, 152]},
    ]
    tracks = {
        1: {
            "first_frame": 0,
            "history_pixels": [(125, 150), (127, 152)],
        }
    }

    metrics = evaluator.evaluate(gt, tracks, total_frames=2)
    assert metrics["mota"] >= 0.90
    assert metrics["id_switches"] == 0


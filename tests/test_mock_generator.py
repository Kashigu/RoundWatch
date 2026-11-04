"""Tests for the synthetic mock data generator."""

import tempfile
from pathlib import Path
import pytest
from src.simulation.mock_generator import MockRoundaboutGenerator, run_generator


def test_mock_generator_coordinate_transformation():
    gen = MockRoundaboutGenerator(fps=20, duration_sec=5.0)

    # Road center (0, 0) should be at origin pixel (width//2, height//2)
    u, v = gen.road_to_pixel(0.0, 0.0)
    assert u == gen.origin_px[0]
    assert v == gen.origin_px[1]

    # Round trip
    x, y = 12.5, -8.2
    u, v = gen.road_to_pixel(x, y)
    rx, ry = gen.pixel_to_road(u, v)

    assert abs(rx - x) < 0.2
    assert abs(ry - y) < 0.2


def test_single_trajectory_generation():
    gen = MockRoundaboutGenerator(fps=20, duration_sec=10.0)
    veh = gen.generate_single_trajectory(
        vehicle_id=1,
        entry_leg=1,
        maneuver="Compliant_Exit1",
        start_time=1.0,
    )

    assert veh["vehicle_id"] == 1
    assert veh["is_compliant"] is True
    assert veh["maneuver"] == "Compliant_Exit1"
    assert len(veh["trajectory"]) >= 15

    pt = veh["trajectory"][0]
    assert "frame_idx" in pt
    assert "timestamp" in pt
    assert "x" in pt
    assert "y" in pt
    assert "bbox" in pt
    assert len(pt["bbox"]) == 4


def test_mock_dataset_and_video_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = run_generator(
            output_dir=tmpdir,
            num_vehicles=5,
            duration_sec=3.0,
            fps=10,
        )

        assert (out_path / "ground_truth_trajectories.json").exists()
        assert (out_path / "synthetic_roundabout_feed.mp4").exists()
        assert (out_path / "metadata.json").exists()
        assert (out_path / "synthetic_roundabout_feed.mp4").stat().st_size > 1000


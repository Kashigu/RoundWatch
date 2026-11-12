"""Capture a 60-second CARLA/SUMO validation clip and export YOLO labels."""

from __future__ import annotations

from pathlib import Path

from scripts.export_yolo_benchmark import export_dataset
from src.simulation.co_simulation_recorder import CarlaSumoCoSimulationRecorder


def main() -> None:
    recording_dir = Path("data/recordings")
    recorder = CarlaSumoCoSimulationRecorder(
        sumo_cfg="data/sumo/roundabout.sumocfg",
        fps=20,
        output_dir=str(recording_dir),
    )
    try:
        recorder.setup()
        recorder.run_recording(duration_sec=60.0)
    finally:
        recorder.cleanup()

    export_dataset(
        video_path=recording_dir / "carla_sumo_cam_south.mp4",
        ground_truth_path=recording_dir / "carla_sumo_gt.json",
        output_dir=Path("data/yolo_benchmark/val"),
        sample_every=20,
        camera="south",
    )


if __name__ == "__main__":
    main()

"""Record a CARLA/SUMO validation clip and export YOLO benchmark datasets.

Each run gets its own folder, so earlier recordings are never overwritten:

    data/recordings/<name>/            videos + carla_sumo_gt.json
    data/yolo_benchmark/<name>/val_<N>fps/   images, labels, dataset.yaml

Labels come from the instance-segmentation camera (exact visible boxes).
CARLA must already be running (scripts/start_carla.bat).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_yolo_benchmark import export_dataset
from scripts.runtime_paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Recording name / folder (default: current date and time)",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to record (default 60)")
    parser.add_argument(
        "--fps",
        type=float,
        nargs="+",
        default=[1.0],
        help="Dataset sampling rates to export, e.g. --fps 1 5 20 (default 1)",
    )
    parser.add_argument("--camera", choices=("south", "north"), default="south")
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Also save the instance-segmentation frames as PNGs (large)",
    )
    parser.add_argument(
        "--projected-labels",
        action="store_true",
        help="Use the old projected 3D-box labels instead of instance segmentation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recording_dir = PROJECT_ROOT / "data" / "recordings" / args.name
    dataset_root = PROJECT_ROOT / "data" / "yolo_benchmark" / args.name
    for path in (recording_dir, dataset_root):
        if path.exists() and any(path.iterdir()):
            raise SystemExit(f"{path} already exists and is not empty; choose another --name")

    # Imported here so --help works without CARLA/SUMO on the path.
    from src.simulation.co_simulation_recorder import CarlaSumoCoSimulationRecorder

    recorder = CarlaSumoCoSimulationRecorder(
        sumo_cfg=str(PROJECT_ROOT / "data" / "sumo" / "roundabout.sumocfg"),
        fps=20,
        output_dir=str(recording_dir),
        instance_labels=not args.projected_labels,
        save_instance_masks=args.save_masks,
    )
    try:
        recorder.setup()
        recorder.run_recording(duration_sec=args.duration)
    finally:
        recorder.cleanup()

    for fps in args.fps:
        label = f"{fps:g}"
        export_dataset(
            video_path=recording_dir / f"carla_sumo_cam_{args.camera}.mp4",
            ground_truth_path=recording_dir / "carla_sumo_gt.json",
            output_dir=dataset_root / f"val_{label}fps",
            camera=args.camera,
            sample_fps=fps,
        )

    print(f"\nRecording: {recording_dir}")
    print(f"Datasets:  {dataset_root}")


if __name__ == "__main__":
    main()

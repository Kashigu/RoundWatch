"""Export CARLA recorder output as a sampled YOLO validation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.runtime_paths import PROJECT_ROOT

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

COCO_CLASS_BY_CARLA_TYPE = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


def class_id_for_vehicle(vehicle_type: str) -> int:
    """Map a CARLA blueprint ID to the corresponding COCO vehicle class."""
    normalized = vehicle_type.lower()
    if "carlacola" in normalized or "truck" in normalized:
        return COCO_CLASS_BY_CARLA_TYPE["truck"]
    if "motorcycle" in normalized:
        return COCO_CLASS_BY_CARLA_TYPE["motorcycle"]
    if "bus" in normalized:
        return COCO_CLASS_BY_CARLA_TYPE["bus"]
    return COCO_CLASS_BY_CARLA_TYPE["car"]


def yolo_annotation(
    class_id: int,
    bbox: list[int] | tuple[int, int, int, int],
    width: int,
    height: int,
) -> str:
    """Convert a pixel box to a normalized YOLO annotation row."""
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid image dimensions or bounding box: {bbox}, {width}x{height}")

    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def estimate_background(video_path: Path, stride: int = 10) -> np.ndarray:
    """Median of every ``stride``-th frame: the static scene with moving vehicles removed."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open recorded video: {video_path}")
    frames = []
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                frames.append(frame)
            frame_idx += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def foreground_mask(frame: np.ndarray, background: np.ndarray, threshold: int = 30) -> np.ndarray:
    """Binary mask of pixels that differ from the static background."""
    diff = cv2.absdiff(frame, background).max(axis=2)
    mask = (diff > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def refine_bbox(
    mask: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    margin_ratio: float = 0.5,
    min_margin: int = 20,
    min_visible_ratio: float = 0.1,
    min_pixels: int = 30,
) -> tuple[int, int, int, int] | None:
    """Snap a projected simulator box onto the visible vehicle pixels.

    The CARLA projection is a loose hull of the 3D box and can lag the camera
    frame by a tick, so the box is re-fitted to the foreground pixels found in
    a search window around it. Returns ``None`` when (almost) nothing of the
    vehicle is visible, e.g. when it is hidden behind the central monument.
    """
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    box_w, box_h = x2 - x1, y2 - y1
    mx = max(min_margin, int(box_w * margin_ratio))
    my = max(min_margin, int(box_h * margin_ratio))
    sx1, sy1 = max(0, x1 - mx), max(0, y1 - my)
    sx2, sy2 = min(width, x2 + mx + 1), min(height, y2 + my + 1)

    window = mask[sy1:sy2, sx1:sx2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(window, connectivity=8)
    # Keep blobs that are big enough and touch the original projected box.
    keep = np.zeros(window.shape, dtype=bool)
    for idx in range(1, count):
        bx, by, bw, bh, area = stats[idx]
        if area < min_pixels:
            continue
        gx1, gy1 = bx + sx1, by + sy1
        gx2, gy2 = gx1 + bw - 1, gy1 + bh - 1
        if gx2 < x1 or gx1 > x2 or gy2 < y1 or gy1 > y2:
            continue
        keep |= labels == idx

    visible = int(keep.sum())
    if visible < max(min_pixels, min_visible_ratio * box_w * box_h):
        return None
    ys, xs = np.nonzero(keep)
    rx1, ry1 = int(xs.min()) + sx1, int(ys.min()) + sy1
    rx2, ry2 = int(xs.max()) + sx1, int(ys.max()) + sy1
    if rx2 - rx1 < 4 or ry2 - ry1 < 4:
        return None
    return rx1, ry1, rx2, ry2


def sampled_frame_indices(total_frames: int, source_fps: float, target_fps: float) -> set[int]:
    """Frames that sample a ``source_fps`` video at ``target_fps``, evenly spaced.

    Picks the first frame of every 1/target_fps interval, so non-divisors such
    as 3 fps from a 20 fps recording still yield exactly 3 frames per second.
    """
    if target_fps <= 0 or target_fps > source_fps:
        raise ValueError(f"target_fps must be in (0, {source_fps}], got {target_fps}")
    picked = set()
    last_slot = -1
    for frame_idx in range(total_frames):
        slot = int(frame_idx * target_fps / source_fps + 1e-9)
        if slot != last_slot:
            picked.add(frame_idx)
            last_slot = slot
    return picked


def export_dataset(
    video_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    sample_every: int = 20,
    camera: str = "south",
    refine: bool | None = None,
    sample_fps: float | None = None,
) -> int:
    """Write sampled frames and matching YOLO labels; return image count.

    Frames are taken every ``sample_every`` frames, or at ``sample_fps``
    frames per second of video when given. With ``refine`` the projected boxes
    are fitted to the visible vehicle pixels (background subtraction) and
    fully occluded vehicles are dropped. ``refine=None`` (default) decides per
    label: boxes from the instance-segmentation camera are already exact and
    used as-is; projected 3D boxes (older recordings, or ticks where the
    segmentation image was missing) are refined.
    """
    if sample_every < 1:
        raise ValueError("sample_every must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory: {output_dir}")

    with ground_truth_path.open(encoding="utf-8") as file:
        records: list[dict[str, Any]] = json.load(file)
    records_by_frame: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        frame_idx = int(record["frame_idx"])
        records_by_frame.setdefault(frame_idx, []).append(record)

    source_key = f"label_source_{camera}"

    def needs_refine(record: dict[str, Any]) -> bool:
        if refine is not None:
            return refine
        return record.get(source_key) != "instance_segmentation"

    labelled_records = [r for r in records if r.get(f"bbox_{camera}") is not None]
    n_refine = sum(1 for r in labelled_records if needs_refine(r))
    if refine is None:
        n_exact = len(labelled_records) - n_refine
        print(
            f"{n_exact} {camera} labels from instance segmentation (used as-is), "
            f"{n_refine} projected boxes (refined against visible pixels)"
        )
        if n_exact and n_refine > 0.05 * len(labelled_records):
            print("WARNING: many ticks lacked segmentation images; consider re-recording.")
    background = estimate_background(video_path) if n_refine else None
    dropped = 0
    labelled = 0

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open recorded video: {video_path}")
    if sample_fps is not None:
        keep_frames = sampled_frame_indices(
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            capture.get(cv2.CAP_PROP_FPS),
            sample_fps,
        )
        is_sampled = keep_frames.__contains__
    else:
        is_sampled = lambda idx: idx % sample_every == 0  # noqa: E731

    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if is_sampled(frame_idx):
                height, width = frame.shape[:2]
                stem = f"frame_{frame_idx:06d}"
                image_path = image_dir / f"{stem}.jpg"
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError(f"Could not write image: {image_path}")

                rows = []
                bbox_key = f"bbox_{camera}"
                mask = foreground_mask(frame, background) if background is not None else None
                for record in records_by_frame.get(frame_idx, []):
                    bbox = record.get(bbox_key)
                    if bbox is None:
                        continue
                    if mask is not None and needs_refine(record):
                        bbox = refine_bbox(mask, bbox)
                        if bbox is None:
                            dropped += 1
                            continue
                    labelled += 1
                    rows.append(
                        yolo_annotation(
                            class_id_for_vehicle(record.get("type", "")),
                            bbox,
                            width,
                            height,
                        )
                    )
                (label_dir / f"{stem}.txt").write_text(
                    "\n".join(rows) + ("\n" if rows else ""),
                    encoding="utf-8",
                )
                exported += 1
            frame_idx += 1
    finally:
        capture.release()

    if exported == 0:
        raise RuntimeError(f"No sampled frames were exported from {video_path}")

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "train: null\n"
        "val: images\n"
        "names:\n"
        + "".join(f"  {index}: {name!r}\n" for index, name in enumerate(COCO_NAMES)),
        encoding="utf-8",
    )
    print(f"Exported {exported} images with {labelled} labels to {output_dir}")
    if n_refine:
        print(f"Dropped {dropped} refined labels with no visible vehicle pixels (occluded)")
    print(f"Dataset config: {dataset_yaml}")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=PROJECT_ROOT / "data/recordings/carla_sumo_cam_south.mp4")
    parser.add_argument("--ground-truth", type=Path, default=PROJECT_ROOT / "data/recordings/carla_sumo_gt.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/yolo_benchmark/val")
    parser.add_argument("--sample-every", type=int, default=20, help="Frames between samples (20 = 1 fps at 20 fps)")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Sample this many frames per second of video (overrides --sample-every)",
    )
    parser.add_argument("--camera", choices=("south", "north"), default="south")
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Keep the recorded boxes exactly as they are",
    )
    refine_group.add_argument(
        "--refine",
        dest="refine",
        action="store_true",
        help="Always fit boxes to visible pixels (default: only for projected-box recordings)",
    )
    parser.set_defaults(refine=None)
    args = parser.parse_args()

    export_dataset(
        args.video,
        args.ground_truth,
        args.output_dir,
        sample_every=args.sample_every,
        camera=args.camera,
        refine=args.refine,
        sample_fps=args.fps,
    )


if __name__ == "__main__":
    main()

import json

import cv2
import numpy as np

from scripts.export_yolo_benchmark import (
    class_id_for_vehicle,
    export_dataset,
    refine_bbox,
    sampled_frame_indices,
    yolo_annotation,
)


def test_class_id_for_carla_vehicle_type():
    assert class_id_for_vehicle("vehicle.carlamotors.carlacola") == 7
    assert class_id_for_vehicle("vehicle.volkswagen.t2") == 2


def test_yolo_annotation_uses_normalized_center_format():
    assert yolo_annotation(2, (10, 20, 50, 80), 100, 100) == "2 0.300000 0.500000 0.400000 0.600000"


def test_export_dataset_samples_frames_and_writes_ground_truth_labels(tmp_path):
    video_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (100, 100),
    )
    for _ in range(3):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps([{
            "frame_idx": 0,
            "type": "vehicle.carlamotors.carlacola",
            "bbox_south": [10, 20, 50, 80],
        }]),
        encoding="utf-8",
    )

    output_dir = tmp_path / "dataset"
    assert export_dataset(
        video_path, ground_truth_path, output_dir, sample_every=2, refine=False
    ) == 2
    assert (output_dir / "images" / "frame_000000.jpg").is_file()
    assert (output_dir / "labels" / "frame_000000.txt").read_text(
        encoding="utf-8"
    ) == "7 0.300000 0.500000 0.400000 0.600000\n"
    assert (output_dir / "labels" / "frame_000002.txt").read_text(encoding="utf-8") == ""
    dataset_yaml = (output_dir / "dataset.yaml").read_text(encoding="utf-8")
    assert "train: null" in dataset_yaml
    assert "val: images" in dataset_yaml


def test_refine_bbox_snaps_offset_box_to_visible_vehicle():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[60:90, 40:100] = 1  # vehicle actually visible here
    # Projected box is too large and shifted right/down relative to the vehicle.
    assert refine_bbox(mask, (50, 70, 120, 110)) == (40, 60, 99, 89)


def test_refine_bbox_drops_fully_occluded_vehicle():
    mask = np.zeros((200, 200), dtype=np.uint8)
    assert refine_bbox(mask, (50, 50, 80, 80)) is None


def test_refine_bbox_ignores_unrelated_blob_in_search_window():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:80, 50:90] = 1
    mask[100:110, 120:135] = 1  # neighbour inside the margin, not touching the box
    assert refine_bbox(mask, (48, 48, 92, 82)) == (50, 50, 89, 79)


def test_sampled_frame_indices_handles_non_divisor_rates():
    picked = sorted(sampled_frame_indices(40, 20, 3))
    assert picked == [0, 7, 14, 20, 27, 34]
    assert sorted(sampled_frame_indices(40, 20, 1)) == [0, 20]
    assert len(sampled_frame_indices(1200, 20, 20)) == 1200


def test_export_uses_instance_segmentation_labels_as_is(tmp_path):
    video_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (100, 100))
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps([{
            "frame_idx": 0,
            "type": "vehicle.audi.tt",
            "bbox_south": [10, 20, 50, 80],
            "label_source_south": "instance_segmentation",
        }]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "dataset"
    # Default (auto) must not run background refinement, which would drop this
    # label because the synthetic video has no visible vehicle.
    export_dataset(video_path, ground_truth_path, output_dir, sample_every=1)
    assert (output_dir / "labels" / "frame_000000.txt").read_text(
        encoding="utf-8"
    ) == "2 0.300000 0.500000 0.400000 0.600000\n"


def test_export_refines_only_projected_labels_in_mixed_recording(tmp_path):
    video_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (100, 100))
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps([
            {"frame_idx": 0, "type": "vehicle.audi.tt", "bbox_south": [10, 20, 50, 80],
             "label_source_south": "instance_segmentation"},
            # Projected box on a tick without segmentation: refined, and since the
            # synthetic video shows no vehicle, dropped as not visible.
            {"frame_idx": 0, "type": "vehicle.audi.tt", "bbox_south": [60, 20, 90, 80],
             "label_source_south": "projected"},
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "dataset"
    export_dataset(video_path, ground_truth_path, output_dir, sample_every=1)
    assert (output_dir / "labels" / "frame_000000.txt").read_text(
        encoding="utf-8"
    ) == "2 0.300000 0.500000 0.400000 0.600000\n"

# Changelog

## [Unreleased] — Exact benchmark labels and multi-model benchmark

### Summary

- **Exact labels.** Benchmark labels now come from CARLA's
  instance-segmentation camera instead of projected 3D bounding boxes. Each box
  covers exactly the visible pixels of one vehicle, and hidden vehicles get no
  label.
- **Timing fix.** The recorder now pairs every video frame with the vehicle
  poses of the same simulation tick. Before, it could pair a frame with the
  previous tick's poses, which shifted the boxes.
- **Recording workflow.** Recordings are captured with
  `record_benchmark.bat`. Each one goes into its own folder, so earlier
  recordings are never overwritten.
- **Benchmark script.** `scripts/run_yolo_benchmark.py` compares YOLOv8 and
  YOLO11 across datasets and image sizes in one run.

### Added

#### `src/simulation/instance_labels.py` (new)
NumPy-only helpers that turn an instance-segmentation frame into label boxes.
They don't import CARLA, so they can be tested without a simulator.
- `decode_instance_image()` reads CARLA's BGRA encoding: semantic tag in red,
  instance ID in green (low byte) and blue (high byte).
- `vehicle_instance_map()` keeps only pixels with vehicle semantic tags (14–19:
  car, truck, bus, train, motorcycle, bicycle).
- `match_vehicle_instances()` assigns each vehicle the instance blob inside its
  projected 3D box, best overlap first, with each blob used once. The label is
  the full extent of that blob. A vehicle gets no label when fewer than 25
  pixels are visible, or when less than half of the blob lies inside its box
  (the blob then belongs to a neighbouring vehicle).
  - Matching uses overlap rather than IDs because, in CARLA 0.9.15, instance
    IDs do not equal actor IDs. A test recording showed 0% agreement.

#### `scripts/run_yolo_benchmark.py` (new)
Validates every combination of dataset × model × image size.
- Accepts a dataset folder, a `dataset.yaml`, or a recording folder; with a
  recording folder it runs every `val_*fps` set, sorted by rate.
- Defaults: `yolov8n yolov8s yolo11n yolo11s` at 1920 px, on the GPU when
  available. Override with `--models`, `--imgsz` and `--device`.
- Missing official weights are downloaded to the project root.
- Writes `summary.json` (all metrics, per class), `summary.md` (per-run table
  plus a model-comparison table) and per-run plots to `runs/benchmark/<name>/`.
- Includes the `if __name__ == "__main__"` guard needed on Windows; without it,
  Ultralytics' data-loader workers re-run the script.

#### `record_benchmark.bat` (new)
Windows launcher for `scripts/capture_yolo_benchmark.py`. It loads `.env`, sets
the SUMO and CARLA paths like `run_live_co_simulation.bat`, and passes all
arguments through.

#### `tests/test_instance_labels.py` (new)
8 tests using synthetic segmentation images:
- ID decoding;
- tight boxes;
- partly and fully hidden vehicles;
- the minimum visible-pixel threshold;
- overlapping vehicles;
- a neighbouring vehicle that is only partly inside the projected box;
- tick-exact frame retrieval in `CameraRig`.

#### `docs/benchmark_results/` (new)
Small JSON/Markdown summaries of every benchmark run so far. They stay in the
repo while `runs/` is ignored.
- `original_recording/`: the numbers behind `docs/yolo_benchmark_report.md`
  (raw vs corrected labels, 640 vs 1920 px, and the 1–20 fps sweep).
- `clear_noon_02/models_by_fps.*`: YOLOv8n/s and YOLO11n/s at 1 and 5 fps with
  exact labels.
- `clear_noon_02/imgsz_sweep_5fps.*`: all four models at 640, 960, 1280, 1600,
  1920 and 2560 px.

### Changed

#### `src/simulation/co_simulation_recorder.py`
- **New constructor options:**
  - `instance_labels=True` mounts an instance-segmentation camera beside each
    RGB camera and takes labels from it;
  - `save_instance_masks=False` optionally writes the segmentation frames as
    PNGs;
  - `min_visible_pixels=25` sets the label threshold.
- **Tick-exact capture.** After each CARLA tick, the recorder reads the
  simulation frame number and waits (up to 10 s) for the RGB and segmentation
  images rendered at that frame. Before, it took whichever image had arrived
  last.
  - The 10 s timeout covers rendering stalls when CARLA loads a vehicle model
    for the first time.
- **Loop reordered.** Each tick now runs: tick → fetch that tick's images →
  record ground truth → apply SUMO vehicle updates for the next tick. Labels
  and frames therefore describe the same world state.
- **Cameras capture every tick** (`sensor_tick=0`). With `sensor_tick=0.05`,
  equal to the world step, floating-point drift made sensors skip ticks at
  random. In the first new-style recording only 37% (south) and 5% (north) of
  ticks had segmentation images.
- **Ground-truth records gain fields:**
  - `bbox_<cam>_projected` (the old-style hull);
  - `visible_px_<cam>`;
  - `label_source_<cam>` (`instance_segmentation` or `projected`).

  Existing fields (`bbox`, `bbox_south`, `bbox_north`, …) keep their meaning.
  Records are still written only for vehicles visible in at least one camera,
  so `src/evaluation/benchmark.py` and other readers are unaffected.
- **Fallback.** If a tick's segmentation image is missing, that camera falls
  back to projected boxes for that tick (at least 10 px per side, as before).
- **Progress reporting.** The progress line shows missing RGB and segmentation
  frame counts. The end-of-run summary reports label counts, skipped hidden
  vehicles, and how often instance IDs matched actor IDs.
- **Cleanup.** `cleanup()` now destroys the co-simulated vehicles. Previously,
  the SUMO bridge's `close()` left them in the CARLA world after every
  recording.

#### `src/simulation/camera_rig.py`
All changes are opt-in; existing callers behave as before.
- `spawn_camera(..., with_instance_segmentation=False, sensor_tick=0.05)`:
  - optionally spawns a `sensor.camera.instance_segmentation` with the same
    resolution, FOV and pose as the RGB camera;
  - makes the capture interval configurable.
- Received images are buffered by simulation frame (last 64). The new
  `wait_for_frame(frame_id)` and `wait_for_instance_frame(frame_id)` return
  the image for an exact tick, or `None` on timeout.
- `compute_2d_bounding_box(actor, min_size=10)` makes the minimum box size
  configurable, so the label matcher can use boxes of any size. It also
  returns `None` for boxes that lie entirely off-screen.
- `destroy()` also removes the segmentation sensor.

#### `scripts/capture_yolo_benchmark.py`
Rewritten as a command-line tool.
- **Options:**
  - `--name` (default: current date and time);
  - `--duration` (default 60 s);
  - `--fps` (one or more export rates, default 1);
  - `--camera south|north`;
  - `--save-masks`;
  - `--projected-labels` (old behaviour, for comparison).
- **Per-recording folders.** Output goes to `data/recordings/<name>/` and
  `data/yolo_benchmark/<name>/val_<N>fps/`. A name that already holds files is
  refused before anything is recorded.
  - Previously the script wrote to `data/recordings/` and
    `data/yolo_benchmark/val`, overwriting the last recording and then failing
    because `val` already existed.
- Paths are resolved from the project root, so the script works from any
  directory, as a script or as `python -m`.

#### `scripts/export_yolo_benchmark.py`
- **Per-label refinement.** `refine` now defaults to `None`, which decides per
  label:
  - boxes with `label_source_<camera> == "instance_segmentation"` are used
    as-is;
  - projected boxes (older recordings, or ticks with a missing segmentation
    image) are refined against the static background as before.
- **CLI.** `--refine` / `--no-refine` (mutually exclusive) force either
  behaviour.
- **Reporting.** The exporter prints how many labels were exact and how many
  were refined. It warns when more than 5% of a new-style recording's labels
  had to fall back.

#### `tests/test_yolo_benchmark.py`
Two new tests:
- exact labels are exported unchanged by default;
- in a mixed recording, only the projected labels are refined.

#### `README.md`
- Section 4 now covers recording with `record_benchmark.bat`, its options,
  the per-recording folder layout, how instance-segmentation labels work, and
  how older recordings are handled.
- New "Running the benchmark" subsection for `scripts/run_yolo_benchmark.py`.

#### `docs/yolo_benchmark_report.md`
"Result files" now points to `docs/benchmark_results/` instead of `runs/`
(which is not committed), and lists the new `clear_noon_02` results.

#### `.gitignore`
Ignores `runs/`, the Ultralytics output folder: plots and prediction images,
about 160 MB so far.

### Notes

- **Output location.** `python -m scripts.capture_yolo_benchmark` still works,
  but its output location changed (see above).
- **Older recordings** (no `label_source_*` fields) are exported exactly as
  before: every box is refined.
- **Re-recording.** `clear_noon_01` was recorded before the `sensor_tick` fix.
  Most of its labels fell back to projected boxes, so it should not be used for
  benchmarks.
- **Close CARLA before benchmarking.** Running both on the 8 GB GPU exhausted
  GPU memory and RAM and slowed validation about 100×. The slow `yolo11s`
  timings at 1920 and 2560 px in `imgsz_sweep_5fps` (92 ms and 323 ms,
  against 24 ms in `models_by_fps`) are probably from memory pressure. Repeat
  that run before relying on those timings.

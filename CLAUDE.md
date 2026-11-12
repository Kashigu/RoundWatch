# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RoundWatch is a roadside roundabout monitor. YOLOv8 finds vehicles in CARLA (Town03) camera feeds, a dual-camera tracker gives each vehicle a persistent ID, trajectories are classified against Portuguese Highway Code Art. 14.º-A (roundabout lane rules), and the results are drawn as AR overlays. Windows-only in practice: the paths, `.bat` launchers and engine locations all assume Windows.

## Environment & commands

All Python runs through the Python 3.10 venv at `.venv310` (CARLA's Python API needs CP310). Run commands from the repo root, because modules use `sys.path.insert(0, os.path.abspath("."))` and relative paths like `configs/...` and `data/...`.

```powershell
.\.venv310\Scripts\python.exe -m pytest tests -v                             # all tests (bare `pytest` also collects scripts/ and scratch/ probes, which need CARLA)
.\.venv310\Scripts\python.exe -m pytest tests/test_perception.py::test_pixel_to_road_ray_projection -v   # single test
.\.venv310\Scripts\python.exe scripts/run_pipeline.py                        # offline end-to-end pipeline on a recorded video
.\run_live_co_simulation.bat                                                 # live CARLA+SUMO window (scripts/watch_live_simulation.py)
.\view_simulation.bat                                                        # AR desktop viewer (scripts/view_live.py)
.\scripts\start_carla.bat                                                    # start the CARLA server (port 2000, 20 FPS)
.\.venv310\Scripts\python.exe scripts/test_carla_connection.py               # check the CARLA client can connect
```

There is no linter or build step configured.

**External engines (not in git):** CARLA 0.9.15 and SUMO 1.28.0 are installed per machine. Their paths come from a gitignored `.env` in the repo root; copy `.env.example` to create it. It sets `CARLA_ROOT`, `SUMO_HOME` and `PYTHON_EXE` (set this to `.venv310\Scripts\python.exe`). Python reads it through `scripts/runtime_paths.py` (`configured_path()`, `configure_sumo_environment()`), and the `.bat` launchers read it through `scripts/load_env.bat`. Variables already set in the process environment take precedence over `.env`. Don't hard-code engine paths. Anything that imports `carla` or `sumo_integration` needs `<CARLA_ROOT>\Co-Simulation\Sumo` and `<SUMO_HOME>\tools` on `sys.path`; the launchers and `runtime_paths` users set this up.

GPU: RTX 50-series cards (sm_120) need PyTorch ≥ 2.7 built for CUDA 12.8 (`pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128`). Older builds make `VehicleDetector` fall back to CPU silently, through the kernel probe in `_resolve_device`.

`requirements.txt` lists only the core packages. torch, ultralytics, traci/sumolib and carla are commented out there but are installed in the venv. Unit tests do not need CARLA or SUMO running. `TrajectoryClassifier` falls back to rule-based evaluation if torch or the model weights are missing.

## Architecture

The data flow runs **simulation → perception → compliance → visualization → evaluation**, one package per stage under `src/`.

- **Coordinate frame:** everything is in CARLA world metres, with the roundabout centre at (0,0). `configs/roundabout.yaml` holds the ring radii: island 14 m, inner lane 14–18 m, outer lane 18–22 m. It also defines the four legs (S=270°, E=0°, N=90°, W=180°) and the infraction rules. `RoundaboutGeometry` (shapely) uses these values to classify points into zones and lanes.
- **Dual camera rig:** two lamp-pole cameras on the south and north splitter islands. Each is at 11.5 m height, pitch −28°, FOV 90°. Their extrinsics appear in `configs/default.yaml` and are **also hard-coded** in `scripts/run_pipeline.py`, `scripts/watch_live_simulation.py` and several scratch scripts. If you change a pose, update every copy. Recorded dual videos are the two panes side by side (south left, north right). `run_pipeline.py` treats any frame wider than 1920 px as dual and splits it at `width // 2`.
- **Projection:** pixel → road uses an exact ray-plane intersection (`pixel_to_road_ray` in `src/perception/tracker.py`) from the bottom-centre of each bbox. `world_to_camera_view` (duplicated in scripts) does the inverse. It uses the CARLA rotation convention, with R = R_pitch @ R_yaw and camera axes (forward, right, up). `RoadwayHomography` plus `configs/homography_town03.json` (`cameras_720` → `cam_south`/`cam_north` H matrices) is an alternative planar mapping.
- **Tracking (two layers):**
  1. *Per camera: the official Ultralytics BoT-SORT.* The project requires BoT-SORT; don't replace it with a custom tracker. `VehicleDetector.track()` calls `model.track(persist=True, tracker="configs/botsort.yaml")`, which gives the Kalman filter, GMC camera-motion compensation and native ReID taken from the YOLO features. Tracker state lives inside the YOLO model, so **each camera needs its own `VehicleDetector` instance**. `BoTSORTTracker` is a thin adapter that feeds detection dicts from any source into the same `ultralytics.trackers.BOTSORT`. It has no native ReID and is mainly used for tests. All BoT-SORT parameters live in `configs/botsort.yaml`.
  2. *Cross-camera fusion: `DynamicMultiCameraTracker`.* It binds each (camera, BoT-SORT local id) pair to one global vehicle ID in road space. A new local track is linked to a vehicle the other camera sees in the same frame (6.5 m gate) or to one recently lost (8/14 m gate). This covers camera hand-over, the central-island occlusion and BoT-SORT ID restarts. Global IDs are sequential and assigned at `min_hits`; vehicles move to `completed_tracks` after `max_age_frames`. It must not use ground truth or preset IDs.
- **Compliance:** `TrajectoryFeatureExtractor` produces a 74-dim feature vector. `ComplianceMLP` maps it to 4 classes (`COMPLIANT_EXIT_1`, `COMPLIANT_EXIT_2PLUS`, `INFRACTION_OUTER_MISUSE`, `INFRACTION_DIRECT_CUTOFF`), each with a Portuguese legal citation. The model is trained by `scripts/train_classifier.py` and its metrics are stored in `models/classifier_metrics.json`. In `run_pipeline.py` the classification is currently dormant: every track is written as COMPLIANT.
- **Maps:** `src/simulation/maps.py` is the registry of co-simulation scenarios. Each `MapSpec` holds the SUMO cfg, the CARLA world (a built-in town, or an `.xodr` that CARLA generates a mesh from via `generate_opendrive_world`), the two camera poses, the `RoundaboutGeometry` parameters and the homography source. Select one with `run_live_co_simulation.bat --map <name>`.
  - `town03`: the default.
  - `roundabout_2lane`: generated by `scripts/build_roundabout_2lane.py`. It has a 2-lane ring at 14–18/18–22 m and 2-lane entries with a solid line between the entry lanes. Exits connect only from the outer lane. Its SUMO vehicle ids encode the ground-truth label (`<leg>_exit<n>_<LABEL>_<k>`). Regenerate the map instead of hand-editing its `.net.xml`/`.xodr`. To check it end to end, run `scripts/verify_roundabout_2lane.py` with a CARLA server running.
  - The OpenDRIVE-generated CARLA world is bare road mesh. `src/simulation/scenery.py` dresses it at runtime with packaged CARLA assets spawned as `static.prop.mesh` (`mesh_path=/Game/Carla/Static/...`, files under `<CARLA_ROOT>/CarlaUE4/Content/Carla/Static`). This adds grass, road markings derived from the SUMO lane shapes, the fountain, islands, street lights, trees and buildings. It runs automatically for maps with `scenery=True`, through `dress_world()`.
  - Gotchas: flat tiles are `Other/SM_Plane` laid with `pitch=-90`; only that face accepts `apply_color_texture_to_objects`, and the other face renders red. Use `SM_*` tree meshes, because `SMF_*` foliage doesn't render as a prop. The Town03 double-arm island light is `Pole/StreetLights/Group3/SM_Streetlight2_group3` plus its `_glass_*` parts. Find what a `BP_*` blueprint is built from with `grep -a -o '/Game/[^ ]*' <bp>.uasset`. Keep `OpendriveGenerationParameters.additional_width=0`: anything larger leaves stepped road edges where roads meet junctions. Make scenery calls through a client with a long timeout; the bridge's own client times out after 2 s.
- **Co-simulation:** SUMO drives traffic (`data/sumo/roundabout.sumocfg` with `Town03.net.xml` and `roundabout.rou.xml`, step 0.05 s). CARLA renders it through CARLA's `sumo_integration` bridge in synchronous mode at 20 FPS. `CarlaSumoCoSimulationRecorder` writes video and trajectories to `data/recordings/`. `MockRoundaboutGenerator` produces synthetic video and ground truth in `data/mock/` for runs without the simulator.
- **Pipeline outputs** go to `output/`: `roundwatch_ar_annotated.mp4`, `tracks.json`, `compliance_results.json` and `benchmark_report.json` (MOTA/IDsw from `src/evaluation/benchmark.py`).

## Repo conventions

- `scripts/` contains real entry points (`run_pipeline.py`, `watch_live_simulation.py`, `view_live.py`, `train_classifier.py`, `calibrate_homography.py`) mixed with many one-off inspect/debug/test probes. `scratch/` contains throwaway experiments, some with hard-coded paths from other machines. Files in either folder named `test_*.py` are **not** pytest tests; the real test suite is `tests/`.
- `data/recordings/`, `output/`, `data/mock/*.mp4` and all `*.pt` weights are gitignored. `yolov8n.pt`/`yolov8s.pt` sit in the repo root, and the pipeline loads `yolov8s.pt` on `cuda:0`.
- Code style: `from __future__ import annotations`, type hints, and module docstrings at the top of each file.

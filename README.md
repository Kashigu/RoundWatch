# RoundWatch: Real-Time Roundabout Monitoring and Rule Compliance System

An intelligent roadside monitoring architecture combining edge-based vehicle detection (YOLO), Multi-Object Tracking under occlusions (BoT-SORT), neural network trajectory compliance analysis (Portuguese Highway Code), and Augmented Reality (AR) visual overlays.

---

## 1. Configure External Tools

External tool locations are machine-specific and belong in a local `.env` file,
not in the source code. Copy the example and edit the paths for your machine:

```powershell
Copy-Item .env.example .env
```

Set `CARLA_ROOT` to the CARLA directory containing `CarlaUE4.exe` and its
`Co-Simulation\Sumo` folder, and `SUMO_HOME` to the SUMO installation directory
containing `tools` and `bin`. `PYTHON_EXE` selects the interpreter used by the
Windows launchers; it defaults to `python` on `PATH`. `SUMO_BINARY` is optional
and only needed if SUMO's executable is not at the standard location under
`SUMO_HOME`.

Use unquoted `KEY=VALUE` entries in `.env`, including for paths with spaces.
Existing process environment variables take precedence over values in the file.
Python scripts load `.env` automatically, and the batch launchers read it as well.
The local `.env` is ignored by Git; do not commit machine-specific paths.

Repository-owned files such as maps and route configs remain relative to the
repository and do not need per-machine configuration. Capture/inspection scripts
write to `output/artifacts` by default; set `ARTIFACT_DIR` to change it. Set
`TRANSCRIPT_FILE` only when using the one-off transcript inspection scripts.

## 2. Toolchain

| Component | Installed Version | Location |
| :--- | :--- | :--- |
| **Python** | Python 3.10+ | Select with `PYTHON_EXE` or the active environment |
| **CARLA Simulator** | 0.9.15 | `CARLA_ROOT` in `.env` |
| **CARLA Python API** | Matching CARLA version | Installed in the selected Python environment (`import carla`) |
| **Eclipse SUMO** | 1.28.0+ | `SUMO_HOME` in `.env` |
| **SUMO Python API** | Matching SUMO version | Installed in the selected Python environment |
| **Town03 SUMO Map** | `Town03.net.xml` | `data\sumo\Town03.net.xml` |

---

## 3. How to Launch & Verify Simulation

### Step 1: Launch CARLA Simulator
Run the launcher script:
```powershell
.\scripts\start_carla.bat
```
The launcher reads `CARLA_ROOT` from `.env`.

### Step 2: Test CARLA Connection
In PowerShell:
```powershell
python scripts/test_carla_connection.py
```

### Step 3: Run Unit Tests
```powershell
python -m pytest -v
```

## 4. YOLO Detection Accuracy Benchmark

With CARLA running (`.\scripts\start_carla.bat`), capture 60 seconds of
CARLA/SUMO traffic and export south-camera datasets with simulator-generated
YOLO labels:

```powershell
.\record_benchmark.bat --name clear_noon_01 --fps 1 5 20
```

Options: `--duration` (seconds, default 60), `--fps` (one or more sampling
rates, default 1), `--camera north`, `--save-masks` (keep segmentation PNGs).
`--name` defaults to the current date and time; an existing non-empty name is
refused, so earlier recordings are never overwritten.

The recording is saved under `data/recordings/<name>`; sampled images, labels,
and `dataset.yaml` are written under `data/yolo_benchmark/<name>/val_<N>fps`.
The labels use COCO
class IDs (car `2`, motorcycle `3`, bus `5`, truck `7`) to match pretrained
YOLOv8 weights.

Labels come from an instance-segmentation camera mounted next to each RGB
camera. It renders the same simulation tick, and its images are only used to
compute labels, never shown to YOLO. Each box covers exactly the visible
pixels of one vehicle. Vehicles that are hidden (e.g. behind the central
monument) or have fewer than 25 visible pixels get no label. The recorder
waits for the images of the exact tick whose vehicle poses it logs, so boxes
and frames stay aligned. Pass `save_instance_masks=True` to
`CarlaSumoCoSimulationRecorder` to also keep the segmentation frames as PNGs.

Older recordings only have projected 3D-box hulls, which are loose, can lag the
video by a tick, and include hidden vehicles. For those, the exporter
automatically fits each box to the visible vehicle pixels (background
subtraction against the median frame of the static camera) and drops labels
with no visible pixels. `--refine` / `--no-refine` on
`scripts/export_yolo_benchmark.py` force either behaviour.

### Running the benchmark

Validate every exported dataset of a recording (GPU if available). By default
the benchmark compares `yolov8n`, `yolov8s`, `yolo11n` and `yolo11s` at
1920 px; missing weights are downloaded to the project root.

```powershell
python scripts/run_yolo_benchmark.py data/yolo_benchmark/clear_noon_02
python scripts/run_yolo_benchmark.py data/yolo_benchmark/clear_noon_02/val_20fps --models yolov8s yolo11s --imgsz 640 1280 1920
```

Results (per-run plots, `summary.json`, and `summary.md` with a per-run table
and a model-comparison table) are written to `runs/benchmark/<name>/`.

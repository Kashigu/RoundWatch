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

With CARLA running, capture 60 seconds of CARLA/SUMO traffic and export one
south-camera frame per second with simulator-generated YOLO labels:

```powershell
python -m scripts.capture_yolo_benchmark
```

The recording is saved under `data/recordings`; sampled images, labels, and
`dataset.yaml` are written under `data/yolo_benchmark/val`. The labels use COCO
class IDs (car `2`, motorcycle `3`, bus `5`, truck `7`) to match pretrained
YOLOv8 weights.

The raw simulator boxes are projected 3D hulls that are loose, can lag the
video by a tick, and include vehicles hidden behind the central monument. By
default the exporter fits each box to the visible vehicle pixels (background
subtraction against the median frame of the static camera) and drops labels
with no visible pixels. Pass `--no-refine` to `scripts/export_yolo_benchmark.py`
to keep the raw projected boxes.

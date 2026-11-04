# RoundWatch: Real-Time Roundabout Monitoring and Rule Compliance System

An intelligent roadside monitoring architecture combining edge-based vehicle detection (YOLO), Multi-Object Tracking under occlusions (BoT-SORT), neural network trajectory compliance analysis (Portuguese Highway Code), and Augmented Reality (AR) visual overlays.

---

## 1. System Directory Architecture

The system keeps application code strictly isolated from large simulation engines and heavy assets:

```
E:\
├── RoundWatch\                      <-- Git Repository / Your Application Code
│   ├── configs/                     <-- Roundabout geometry & camera configs
│   │   ├── default.yaml
│   │   └── roundabout.yaml
│   ├── src/
│   │   ├── simulation/              <-- CARLA-SUMO synchronization & mock generator
│   │   ├── perception/              <-- YOLO & BoT-SORT pipeline
│   │   ├── compliance/              <-- Trajectory analysis & rule verification
│   │   ├── visualization/           <-- AR overlays & interactive player
│   │   └── evaluation/              <-- MOTA, IDsw, and ablation benchmarks
│   ├── data/
│   │   ├── sumo/                    <-- SUMO Town03.net.xml & route files
│   │   ├── mock/                    <-- Synthetic video feeds & ground-truth logs
│   │   └── recordings/              <-- Live CARLA video streams & trajectories
│   ├── scripts/
│   │   ├── start_carla.bat          <-- Quick launcher for CARLA UE4 Server
│   │   └── test_carla_connection.py <-- Verification script for CARLA client
│   └── .venv310/                    <-- Python 3.10 virtual environment
│
└── Simulation_Engines\              <-- Dedicated Simulator Directory (Non-Git)
    ├── CARLA_0.9.15\                <-- Full CARLA 0.9.15 simulator (CarlaUE4.exe)
    ├── SUMO\                        <-- Directory junction to SUMO 1.28.0
    └── downloads\                   <-- Archive staging directory
```

---

## 2. Toolchain Status & Installed Versions

| Component | Installed Version | Location |
| :--- | :--- | :--- |
| **Python** | **Python 3.10.11 (64-bit)** | System (`py -3.10`) & `E:\RoundWatch\.venv310` |
| **CARLA Simulator** | **0.9.15** | `E:\Simulation_Engines\CARLA_0.9.15\CarlaUE4.exe` |
| **CARLA Python API** | **0.9.16 (CP310)** | Installed in `E:\RoundWatch\.venv310` (`import carla`) |
| **Eclipse SUMO** | **1.28.0 (64-bit)** | `C:\Program Files (x86)\SUMO\` (Linked at `E:\Simulation_Engines\SUMO`) |
| **SUMO Python API** | **traci 1.28.0, sumolib 1.28.0** | Installed in `E:\RoundWatch\.venv310` |
| **Town03 SUMO Map** | **Town03.net.xml** | `E:\RoundWatch\data\sumo\Town03.net.xml` (converted from Town03.xodr) |

---

## 3. How to Launch & Verify Simulation

### Step 1: Launch CARLA Simulator
Run the launcher script:
```powershell
.\scripts\start_carla.bat
```
*(Or launch `E:\Simulation_Engines\CARLA_0.9.15\CarlaUE4.exe` directly).*

### Step 2: Test CARLA Connection
In PowerShell:
```powershell
.\.venv310\Scripts\python.exe scripts/test_carla_connection.py
```

### Step 3: Run Unit Tests
```powershell
.\.venv310\Scripts\python.exe -m pytest -v
```

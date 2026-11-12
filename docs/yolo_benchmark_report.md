# YOLOv8 Detection Benchmark — CARLA/SUMO Roundabout

Accuracy and speed of pretrained YOLOv8 (COCO weights, no fine-tuning) at detecting
vehicles in the simulated Town03 roundabout. The test varies the label quality, the
model size, the inference image size, and the frame sampling rate.

## Summary

- **The simulator's raw boxes understated accuracy.** Correcting them raised mAP50 from
  0.44 to 0.63 for `yolov8n` at 1920 px with no other change. mAP50-95 more than
  doubled (0.14 → 0.33).
- **Image size matters more than anything else we tested.** At 640 px, most small and
  distant vehicles are missed. At 1920 px (the pipeline's actual setting), mAP50 rises
  by 0.24–0.30.
- **`yolov8s` beats `yolov8n`.** With all 1200 frames scored, mAP50 is 0.664 vs 0.636,
  and the gap is mostly in cars (0.79 vs 0.71).
- **The sampling rate doesn't change the result.** From 1 to 20 fps, scores stay within
  about ±0.03, and 5 fps or more is within about 0.02 of the full 20 fps result. The
  1 fps test was slightly optimistic for `yolov8s` (0.693 vs 0.664).
- **The GPU makes real-time use feasible.** On the RTX 3060 Ti, `yolov8s` at 1920 px
  takes about 23 ms per frame for inference. That is 14× faster than the CPU and well
  under the 50 ms budget for 20 fps.
- **Recall is the main weakness (about 50–60%).** Almost all misses are tiny vehicles
  on the far road behind the roundabout.

## Setup

| Item | Value |
|---|---|
| Scene | CARLA 0.9.15 Town03 roundabout, SUMO-driven traffic, ClearNoon weather |
| Camera | South pole camera, 1920×1080, 11.5 m high, pitch −28° |
| Clip | 60 s at 20 fps (1200 frames), `data/recordings/carla_sumo_cam_south.mp4` |
| Models | `yolov8n.pt`, `yolov8s.pt` (COCO-pretrained) |
| Classes scored | `car` (COCO 2), `truck` (COCO 7) — no buses or motorcycles appear in the clip |
| Validation | Ultralytics 8.4.208 `model.val()`, default conf 0.001 / IoU 0.7 |
| Hardware | Intel i7-12700K, 16 GB RAM, NVIDIA RTX 3060 Ti 8 GB |
| Software | Python 3.10, torch 2.14.1+cu132 |

P and R in the tables are the precision and recall at the confidence threshold that
gives the best F1, as Ultralytics reports them.

## 1. Effect of label correction

The simulator writes each label by projecting the vehicle's 3D bounding box into the
camera. Those boxes have three problems:

1. They are **loose**, because the 2D hull of a 3D box is larger than the visible car.
2. They are **shifted** relative to the car, because the recorder pairs each video frame
   with vehicle poses from a slightly different simulation tick.
3. They include **vehicles that cannot be seen**, chiefly at the spawn point behind the
   central monument.

The exporter (`scripts/export_yolo_benchmark.py`) now fixes these by default. It builds
an empty-scene background as the median of the clip's frames, since the camera is
static. It then fits each box to the vehicle pixels visible around it and drops a label
when no vehicle pixels are visible.

**Label counts at 1 fps (60 frames)**

| Labels | car | truck | total |
|---|---|---|---|
| Raw projected | 119 | 12 | 131 |
| Corrected | 107 | 11 | 118 |
| Dropped as not visible | — | — | 13 |

**Accuracy at 1 fps, 1920 px**

| Model | Labels | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| yolov8n | raw | 0.506 | 0.449 | 0.440 | 0.142 |
| yolov8n | corrected | 0.615 | 0.604 | **0.627** | **0.332** |
| yolov8s | raw | 0.598 | 0.431 | 0.455 | 0.149 |
| yolov8s | corrected | 0.747 | 0.598 | **0.693** | **0.351** |

mAP50-95 more than doubles. It weights how tightly a box fits, so it benefits most from
correcting the shifted and oversized boxes.

## 2. Effect of image size

Corrected and raw labels at 1 fps (60 frames). The pipeline's detector
(`src/perception/detector.py`) runs at full frame width, i.e. 1920 px.

| Model | Labels | imgsz | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| yolov8n | raw | 640 | 0.524 | 0.196 | 0.212 | 0.073 |
| yolov8n | raw | 1920 | 0.506 | 0.449 | 0.440 | 0.142 |
| yolov8n | corrected | 640 | 0.634 | 0.249 | 0.324 | 0.209 |
| yolov8n | corrected | 1920 | 0.615 | 0.604 | **0.627** | **0.332** |
| yolov8s | raw | 640 | 0.477 | 0.355 | 0.279 | 0.077 |
| yolov8s | raw | 1920 | 0.598 | 0.431 | 0.455 | 0.149 |
| yolov8s | corrected | 640 | 0.594 | 0.442 | 0.449 | 0.259 |
| yolov8s | corrected | 1920 | 0.747 | 0.598 | **0.693** | **0.351** |

**Per class, corrected labels**

| Model | imgsz | car P | car R | car mAP50 | truck P | truck R | truck mAP50 |
|---|---|---|---|---|---|---|---|
| yolov8n | 640 | 0.862 | 0.224 | 0.375 | 0.405 | 0.273 | 0.273 |
| yolov8n | 1920 | 0.860 | 0.515 | 0.680 | 0.369 | 0.693 | 0.574 |
| yolov8s | 640 | 0.979 | 0.430 | 0.683 | 0.210 | 0.455 | 0.214 |
| yolov8s | 1920 | 0.952 | 0.559 | 0.796 | 0.542 | 0.636 | 0.590 |

Downscaling a 1920×1080 frame to 640 px makes distant cars only a few pixels wide.
Precision barely changes, but recall roughly halves, so the model misses vehicles
rather than inventing them.

## 3. Effect of frame sampling rate

Each rate takes evenly spaced frames from the same 60 s clip, using corrected labels at
1920 px on the GPU. The 1 fps rows reproduce the earlier CPU results.

**Dataset sizes**

| fps | Images | car labels | truck labels |
|---|---|---|---|
| 1 | 60 | 107 | 11 |
| 2 | 120 | 215 | 23 |
| 3 | 180 | 320 | 34 |
| 5 | 300 | 537 | 58 |
| 10 | 600 | 1075 | 115 |
| 20 | 1200 | 2152 | 229 |

**Overall accuracy**

| fps | v8n P | v8n R | v8n mAP50 | v8n mAP50-95 | v8s P | v8s R | v8s mAP50 | v8s mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.614 | 0.604 | 0.627 | 0.332 | 0.747 | 0.598 | 0.693 | 0.351 |
| 2 | 0.648 | 0.552 | 0.625 | 0.319 | 0.754 | 0.571 | 0.690 | 0.336 |
| 3 | 0.631 | 0.559 | 0.631 | 0.330 | 0.736 | 0.560 | 0.659 | 0.315 |
| 5 | 0.653 | 0.506 | 0.626 | 0.318 | 0.671 | 0.558 | 0.641 | 0.309 |
| 10 | 0.674 | 0.515 | 0.634 | 0.319 | 0.740 | 0.542 | 0.662 | 0.316 |
| 20 | 0.679 | 0.512 | **0.636** | 0.320 | 0.736 | 0.550 | **0.664** | 0.316 |

**Per-class mAP50**

| fps | v8n car | v8n truck | v8s car | v8s truck |
|---|---|---|---|---|
| 1 | 0.680 | 0.574 | 0.796 | 0.590 |
| 2 | 0.712 | 0.538 | 0.787 | 0.593 |
| 3 | 0.703 | 0.559 | 0.796 | 0.522 |
| 5 | 0.710 | 0.542 | 0.791 | 0.491 |
| 10 | 0.717 | 0.551 | 0.793 | 0.531 |
| 20 | 0.712 | 0.559 | 0.793 | 0.535 |

**Per-class recall**

| fps | v8n car | v8n truck | v8s car | v8s truck |
|---|---|---|---|---|
| 1 | 0.514 | 0.693 | 0.559 | 0.636 |
| 2 | 0.538 | 0.565 | 0.533 | 0.609 |
| 3 | 0.529 | 0.588 | 0.562 | 0.559 |
| 5 | 0.495 | 0.517 | 0.582 | 0.534 |
| 10 | 0.491 | 0.539 | 0.545 | 0.539 |
| 20 | 0.482 | 0.541 | 0.547 | 0.553 |

What the rate comparison shows:

- **Car scores are stable at every rate** (`yolov8s` ≈ 0.79, `yolov8n` ≈ 0.71). Nearly
  all the movement comes from trucks, which have only 11 labels at 1 fps.
- **Extra frames reduce noise but add no new information.** Neighbouring frames of one
  clip are near-duplicates. To learn more about the model, record more or different
  clips (other cameras, weather, traffic density) rather than sample one clip more
  densely.
- **Use 20 fps for headline numbers.** Use 5 fps or more for quick checks; it is within
  about 0.02 mAP50 of 20 fps.

## 4. Speed and hardware

**Single-frame latency, `yolov8s` at 1920 px** (20 runs after warm-up, one image)

| Device | ms / frame | Throughput |
|---|---|---|
| CPU (i7-12700K) | 386.7 | ≈ 2.6 fps |
| GPU (RTX 3060 Ti) | 27.1 | ≈ 37 fps |

**GPU validation timing at 1920 px, batches of 8** (averaged over the 1200-frame,
20 fps run)

| Model | Inference ms / frame | Total ms / frame (incl. pre/post-processing) | Wall time, 1200 frames |
|---|---|---|---|
| yolov8n | 10.9 | 27.2 | 62 s |
| yolov8s | 22.9 | 39.6 | 76 s |

Live processing at 20 fps allows 50 ms per frame, and both models fit within it on the
GPU. `yolov8s` costs about 12 ms more per frame than `yolov8n` and is the better choice
on this hardware.

> Note: until this benchmark the project environment had the CPU-only PyTorch build,
> so every detector run used the CPU even when `cuda:0` was requested. Install the
> CUDA build with
> `pip install torch==2.14.1 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132`.

## 5. Limitations

- **One clip.** All results come from a single 60 s recording from one camera in clear
  daylight. They don't cover other viewpoints, weather, night, or heavy traffic.
- **The label correction is heuristic.** The fix uses background subtraction, so a
  vehicle that stayed parked for most of the clip would merge into the background and
  be dropped. That does not happen in this recording. Shadows and adjacent vehicles can
  occasionally widen a box. The proper fix is to generate boxes in the recorder from
  CARLA's instance-segmentation camera and to correct the frame/pose timing offset in
  `src/simulation/co_simulation_recorder.py`.
- **Few trucks.** There are 11–229 truck labels from a handful of physical trucks, so
  truck metrics are noisy.
- **No fine-tuning.** Both models use stock COCO weights. Fine-tuning on simulator
  frames, or using SAHI-style tiling for small objects, is the obvious next step for
  recall.
- **Other model families were not measured.** YOLO11 and YOLO26 were not compared.

## Reproducing

```powershell
# Export a corrected dataset at N frames per second
python scripts/export_yolo_benchmark.py --fps 5 --output-dir data/yolo_benchmark/fps/val_5fps

# Raw simulator boxes instead (for comparison)
python scripts/export_yolo_benchmark.py --fps 1 --no-refine --output-dir data/yolo_benchmark/val

# Validate (GPU, full resolution)
yolo val model=yolov8s.pt data=data/yolo_benchmark/fps/val_5fps/dataset.yaml imgsz=1920 device=0
```

**Result files**

- Section 1–2 runs: `runs/detect/bench_1fps_*`; numbers in
  `runs/detect/bench_1fps_summary_{640,1920}.json`
- Section 3 runs: `runs/detect/fps_sweep/*`; numbers in
  `runs/detect/fps_sweep/summary.json`
- The original run being re-assessed: `runs/detect/val` (`yolov8n`, raw labels, 1920 px)

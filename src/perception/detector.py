"""YOLO Vehicle Detector Wrapper for RoundWatch.

Supports YOLOv8 / YOLOv11 models running on CUDA GPU,
filtering relevant vehicle classes (cars, buses, trucks, motorcycles).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    YOLO = None
    torch = None


def _resolve_device(device: str) -> str:
    """Fall back to CPU when the requested CUDA device is unavailable."""
    cuda_requested = device.startswith("cuda") or device.isdigit()
    if not cuda_requested:
        return device

    if torch is None or not torch.cuda.is_available():
        warnings.warn(
            f"CUDA device {device!r} was requested but CUDA is unavailable; using CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    if device.startswith("cuda:"):
        try:
            device_index = int(device.partition(":")[2])
        except ValueError:
            return device
        if device_index >= torch.cuda.device_count():
            warnings.warn(
                f"CUDA device {device!r} was requested but only "
                f"{torch.cuda.device_count()} device(s) are available; using CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"

    # CUDA can report as available while the PyTorch build lacks kernels for this GPU
    # (e.g. RTX 50-series / sm_120 needs PyTorch >= 2.7 built for CUDA 12.8).
    try:
        torch.zeros(1, device=device if device.startswith("cuda") else f"cuda:{device}")
    except Exception as exc:  # noqa: BLE001 - any failure means the GPU is unusable here
        warnings.warn(
            f"CUDA device {device!r} is not usable with this PyTorch build ({exc}); using CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    return device


class VehicleDetector:
    """Wrapper around Ultralytics YOLO for real-time edge vehicle detection."""

    # COCO Class mapping: 2: car, 3: motorcycle, 5: bus, 7: truck
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        confidence_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: str = "cuda:0",
        imgsz: Optional[int] = None,
        agnostic_nms: bool = True,
    ) -> None:
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.device = _resolve_device(device)
        self.model_name = model_name
        # One box per vehicle: without this a van/truck is often boxed as both 'car' and 'truck',
        # which becomes two tracks and duplicate global IDs downstream.
        self.agnostic_nms = agnostic_nms
        self.imgsz = imgsz  # YOLO inference size; None = max(1280, frame size)

        if YOLO is not None:
            self.model = YOLO(model_name)
        else:
            self.model = None

    def detect(
        self, frame: np.ndarray, target_classes: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Run inference on an RGB/BGR image frame and return vehicle detections."""
        if self.model is None:
            raise RuntimeError("Ultralytics YOLO is not installed.")

        if target_classes is None:
            target_classes = list(self.VEHICLE_CLASSES.keys())

        # Perform inference on GPU with high-res feature preservation
        h, w = frame.shape[:2]
        img_size = self.imgsz or max(1280, w, h)

        results = self.model.predict(
            source=frame,
            conf=self.confidence_thresh,
            iou=self.iou_thresh,
            imgsz=img_size,
            classes=target_classes,
            device=self.device,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )

        return self._parse_results(results, with_ids=False)

    def track(
        self,
        frame: np.ndarray,
        tracker_cfg: str = "configs/botsort.yaml",
        target_classes: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Detect and track vehicles in one camera stream with the official Ultralytics BoT-SORT.

        Tracker state (Kalman filter, GMC camera-motion compensation, ReID features) persists across
        calls, so each camera stream must use its own VehicleDetector instance.
        Returns confirmed tracks only, each with a per-camera ``track_id``.
        """
        if self.model is None:
            raise RuntimeError("Ultralytics YOLO is not installed.")

        if target_classes is None:
            target_classes = list(self.VEHICLE_CLASSES.keys())

        h, w = frame.shape[:2]
        img_size = self.imgsz or max(1280, w, h)

        # conf=None -> Ultralytics passes low-confidence boxes (0.1) to BoT-SORT's second association stage
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=tracker_cfg,
            iou=self.iou_thresh,
            imgsz=img_size,
            classes=target_classes,
            device=self.device,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )
        return self._parse_results(results, with_ids=True)

    def _parse_results(self, results: Any, with_ids: bool) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if with_ids and boxes.id is None:
            return detections

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()

            x1 = int(round(xyxy[0]))
            y1 = int(round(xyxy[1]))
            x2 = int(round(xyxy[2]))
            y2 = int(round(xyxy[3]))

            # Filter tiny noise (< 12px)
            if (x2 - x1) < 12 or (y2 - y1) < 12:
                continue

            det = {
                "bbox": (x1, y1, x2, y2),
                "class_id": cls_id,
                "class_name": self.VEHICLE_CLASSES.get(cls_id, "car"),
                "conf": round(conf, 3),
            }
            if with_ids:
                det["track_id"] = int(boxes.id[i].item())
            detections.append(det)

        return detections

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
    ) -> None:
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.device = _resolve_device(device)
        self.model_name = model_name

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
        img_size = max(w, h)
        if img_size < 1280:
            img_size = 1280

        results = self.model.predict(
            source=frame,
            conf=self.confidence_thresh,
            iou=self.iou_thresh,
            imgsz=img_size,
            classes=target_classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        r = results[0]
        boxes = r.boxes

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

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_id": cls_id,
                "class_name": self.VEHICLE_CLASSES.get(cls_id, "car"),
                "conf": round(conf, 3),
            })

        return detections

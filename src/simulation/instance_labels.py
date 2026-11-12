"""Exact 2D vehicle boxes from CARLA's instance-segmentation camera.

CARLA 0.9.14+ ``sensor.camera.instance_segmentation`` renders, for every pixel,
the semantic tag in the red channel and a per-object instance ID in the green
(low byte) and blue (high byte) channels. Raw image data arrives as BGRA.

Instance IDs are not guaranteed to equal CARLA actor IDs across versions, so
each vehicle is matched to the instance blob that covers most of its projected
3D bounding box. The resulting box is the extent of the pixels that are
actually visible: tight, occlusion-aware, and empty when the vehicle is hidden.
This module is NumPy-only so it can be tested without a running simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np

# CityScapes-aligned semantic tags used by CARLA 0.9.14+.
VEHICLE_SEMANTIC_TAGS = (14, 15, 16, 17, 18, 19)  # car, truck, bus, train, motorcycle, bicycle

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class InstanceLabel:
    """Visible box of one vehicle in one camera."""

    bbox: Box
    visible_pixels: int
    instance_id: int


def decode_instance_image(bgra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a raw BGRA instance-segmentation frame into (semantic tags, instance IDs)."""
    if bgra.ndim != 3 or bgra.shape[2] < 3:
        raise ValueError(f"Expected an HxWx3/4 image, got shape {bgra.shape}")
    tags = bgra[:, :, 2].astype(np.int32)
    ids = bgra[:, :, 1].astype(np.int32) | (bgra[:, :, 0].astype(np.int32) << 8)
    return tags, ids


def vehicle_instance_map(
    bgra: np.ndarray,
    vehicle_tags: Sequence[int] = VEHICLE_SEMANTIC_TAGS,
) -> np.ndarray:
    """Instance-ID image where non-vehicle pixels are -1."""
    tags, ids = decode_instance_image(bgra)
    return np.where(np.isin(tags, vehicle_tags), ids, -1)


def _instance_extents(instance_map: np.ndarray) -> dict[int, tuple[Box, int]]:
    """Full-image (box, pixel count) for every vehicle instance ID."""
    ys, xs = np.nonzero(instance_map >= 0)
    if ys.size == 0:
        return {}
    ids = instance_map[ys, xs]
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    x1 = np.full(unique.size, np.iinfo(np.int64).max)
    y1 = np.full(unique.size, np.iinfo(np.int64).max)
    x2 = np.full(unique.size, -1)
    y2 = np.full(unique.size, -1)
    np.minimum.at(x1, inverse, xs)
    np.minimum.at(y1, inverse, ys)
    np.maximum.at(x2, inverse, xs)
    np.maximum.at(y2, inverse, ys)
    return {
        int(uid): ((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])), int(counts[i]))
        for i, uid in enumerate(unique)
    }


def match_vehicle_instances(
    instance_map: np.ndarray,
    projected_boxes: Mapping[Hashable, Box | None],
    min_visible_pixels: int = 25,
    min_overlap_ratio: float = 0.5,
    search_margin: int = 4,
) -> dict[Hashable, InstanceLabel]:
    """Assign each vehicle the instance blob inside its projected 3D box.

    ``projected_boxes`` maps a vehicle key (e.g. actor ID) to the image-space
    hull of its 3D bounding box, which must come from the same simulation
    tick as ``instance_map``. Each instance is assigned to at most one vehicle,
    best overlap first. A vehicle gets no label when fewer than
    ``min_visible_pixels`` of it are visible, or when less than
    ``min_overlap_ratio`` of the candidate blob lies inside its projected box
    (the blob then belongs to a different, overlapping vehicle).
    """
    height, width = instance_map.shape[:2]
    extents = _instance_extents(instance_map)
    if not extents:
        return {}

    candidates: list[tuple[int, Hashable, int]] = []
    for key, box in projected_boxes.items():
        if box is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1 - search_margin), max(0, y1 - search_margin)
        x2, y2 = min(width - 1, x2 + search_margin), min(height - 1, y2 + search_margin)
        if x2 < x1 or y2 < y1:
            continue
        crop = instance_map[y1:y2 + 1, x1:x2 + 1]
        ids, counts = np.unique(crop[crop >= 0], return_counts=True)
        for inst_id, inside in zip(ids.tolist(), counts.tolist()):
            total = extents[inst_id][1]
            if inside >= min_visible_pixels and inside / total >= min_overlap_ratio:
                candidates.append((inside, key, inst_id))

    labels: dict[Hashable, InstanceLabel] = {}
    used: set[int] = set()
    for inside, key, inst_id in sorted(candidates, key=lambda c: c[0], reverse=True):
        if key in labels or inst_id in used:
            continue
        bbox, pixels = extents[inst_id]
        labels[key] = InstanceLabel(bbox=bbox, visible_pixels=pixels, instance_id=inst_id)
        used.add(inst_id)
    return labels

import threading

import numpy as np

from src.simulation.camera_rig import CameraRig
from src.simulation.instance_labels import (
    decode_instance_image,
    match_vehicle_instances,
    vehicle_instance_map,
)

CAR, BUILDING = 14, 3


def paint(img, x1, y1, x2, y2, tag, instance_id=0):
    """Fill an inclusive rectangle of a BGRA instance-segmentation image."""
    img[y1:y2 + 1, x1:x2 + 1, 2] = tag
    img[y1:y2 + 1, x1:x2 + 1, 1] = instance_id & 0xFF
    img[y1:y2 + 1, x1:x2 + 1, 0] = (instance_id >> 8) & 0xFF


def blank(h=200, w=300):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 2] = 1  # road everywhere
    return img


def test_decode_uses_green_low_byte_and_blue_high_byte():
    img = blank(2, 2)
    paint(img, 0, 0, 0, 0, CAR, instance_id=0x1234)
    tags, ids = decode_instance_image(img)
    assert tags[0, 0] == CAR
    assert ids[0, 0] == 0x1234


def test_box_is_extent_of_visible_pixels_not_projection():
    img = blank()
    paint(img, 50, 60, 99, 89, CAR, instance_id=7)
    labels = match_vehicle_instances(vehicle_instance_map(img), {"a": (40, 50, 120, 110)})
    assert labels["a"].bbox == (50, 60, 99, 89)
    assert labels["a"].visible_pixels == 50 * 30


def test_partially_occluded_vehicle_gets_box_of_visible_part():
    img = blank()
    paint(img, 50, 60, 99, 89, CAR, instance_id=7)
    paint(img, 75, 40, 120, 120, BUILDING)  # monument covers the right half
    labels = match_vehicle_instances(vehicle_instance_map(img), {"a": (50, 60, 99, 89)})
    assert labels["a"].bbox == (50, 60, 74, 89)


def test_fully_hidden_vehicle_gets_no_label():
    img = blank()
    paint(img, 40, 40, 140, 140, BUILDING)
    labels = match_vehicle_instances(vehicle_instance_map(img), {"hidden": (60, 60, 90, 90)})
    assert labels == {}


def test_tiny_visible_sliver_below_threshold_is_dropped():
    img = blank()
    paint(img, 60, 60, 63, 63, CAR, instance_id=3)  # 16 px
    labels = match_vehicle_instances(
        vehicle_instance_map(img), {"a": (55, 55, 70, 70)}, min_visible_pixels=25
    )
    assert labels == {}


def test_overlapping_vehicles_each_get_their_own_blob():
    img = blank()
    paint(img, 100, 50, 179, 99, CAR, instance_id=2)  # far car (behind)
    paint(img, 60, 70, 129, 129, CAR, instance_id=1)  # near car, drawn on top
    projected = {"near": (60, 70, 129, 129), "far": (100, 50, 179, 99)}
    labels = match_vehicle_instances(vehicle_instance_map(img), projected)
    assert labels["near"].instance_id == 1
    assert labels["near"].bbox == (60, 70, 129, 129)
    assert labels["far"].instance_id == 2
    assert labels["far"].bbox == (100, 50, 179, 99)


def test_neighbour_mostly_outside_projection_is_not_stolen():
    img = blank()
    paint(img, 100, 50, 199, 99, CAR, instance_id=2)  # neighbour, only its edge inside
    labels = match_vehicle_instances(vehicle_instance_map(img), {"hidden": (60, 50, 109, 99)})
    assert labels == {}


def test_camera_rig_returns_image_for_exact_frame():
    rig = CameraRig(world=None, image_size=(4, 4))
    old = np.zeros((4, 4, 3), np.uint8)
    new = np.ones((4, 4, 3), np.uint8)
    rig._store(rig._rgb_by_frame, 10, old)
    threading.Timer(0.05, rig._store, args=(rig._rgb_by_frame, 11, new)).start()
    assert rig.wait_for_frame(11, timeout=2.0) is new
    assert rig.wait_for_frame(10, timeout=0.1) is old
    assert rig.wait_for_frame(12, timeout=0.05) is None

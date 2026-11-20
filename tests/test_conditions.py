"""Tests for the recording conditions (wind-driven camera sway)."""

from __future__ import annotations

import numpy as np

from src.simulation.conditions import PoleSway, WEATHER_PRESETS


def test_zero_amplitude_is_static():
    sway = PoleSway(amplitude_deg=0.0)
    assert sway.offsets(3.2) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_sway_is_bounded_deterministic_and_moving():
    a = PoleSway(amplitude_deg=0.5, seed=3)
    b = PoleSway(amplitude_deg=0.5, seed=3)
    ts = np.arange(0, 60, 0.05)
    oa = np.array([a.offsets(t) for t in ts])
    ob = np.array([b.offsets(t) for t in ts])
    assert np.allclose(oa, ob)
    rot = oa[:, :3]
    assert np.abs(rot).max() < 0.5 * 1.6       # peak stays near the requested amplitude
    assert rot[:, 0].std() > 0.05              # the camera actually moves
    assert np.abs(oa[:, 3:5]).max() < 0.2      # pole-top displacement is centimetres, not metres
    # Dominant motion near the pole's natural frequency (~1.1 Hz)
    spec = np.abs(np.fft.rfft(rot[:, 0] - rot[:, 0].mean()))
    freqs = np.fft.rfftfreq(len(ts), d=0.05)
    assert 0.7 < freqs[spec.argmax()] < 1.5


def test_custom_presets_have_rain_and_wind_fields():
    storm = WEATHER_PRESETS["Storm"]
    assert storm["precipitation"] > 50 and storm["wind_intensity"] > 50


def test_shake_moves_image_and_labels_together():
    from src.simulation.conditions import shake_frame

    img = np.zeros((360, 640, 3), np.uint8)
    img[150:190, 300:360] = 255  # a white "vehicle"
    sway = PoleSway(amplitude_deg=1.0, seed=1)
    t = next(t for t in np.arange(0, 10, 0.05) if abs(sway.offsets(t)[0]) > 0.5)
    out, boxes = shake_frame(img, [(2, (300, 150, 359, 189))], sway, float(t))
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0][1]
    assert (x1, y1) != (300, 150)                       # the camera moved
    ys, xs = np.nonzero(out[:, :, 0] > 128)
    assert abs(xs.min() - x1) <= 3 and abs(xs.max() - x2) <= 3
    assert abs(ys.min() - y1) <= 3 and abs(ys.max() - y2) <= 3

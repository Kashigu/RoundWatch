"""Environmental test conditions for recordings: weather presets and wind-driven camera shake.

``WEATHER_PRESETS`` maps a name to CARLA ``WeatherParameters`` fields. Names that match a
built-in CARLA preset (``HardRainNoon``, ``WetCloudyNoon``...) can also be used directly.

``PoleSway`` models a camera on top of a lamp pole moving in the wind. A pole bends
in its first mode, at a natural frequency around 1 Hz. Gusts make the motion stronger and
weaker over a few seconds, and there is a little high-frequency vibration on top. The model
gives small rotation offsets (pitch/yaw/roll, degrees) plus the matching displacement of the
pole top (metres) for any time ``t``. It is deterministic for a given seed, so a recording
can be reproduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

# Custom presets (CARLA WeatherParameters field names). Built-in CARLA presets are also accepted.
WEATHER_PRESETS: Dict[str, Dict[str, float]] = {
    "Storm": {
        "cloudiness": 95.0,
        "precipitation": 90.0,
        "precipitation_deposits": 80.0,
        "wind_intensity": 100.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 35.0,
        "fog_density": 15.0,
        "fog_distance": 40.0,
        "wetness": 90.0,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.05,
    },
    "HeavyRain": {
        "cloudiness": 100.0,
        "precipitation": 100.0,
        "precipitation_deposits": 90.0,
        "wind_intensity": 60.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 25.0,
        "fog_density": 8.0,
        "fog_distance": 60.0,
        "wetness": 100.0,
    },
    "Dusk": {
        "cloudiness": 30.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 10.0,
        "sun_azimuth_angle": 270.0,
        "sun_altitude_angle": 4.0,  # just above the horizon; CARLA is already near-night below 0
        "fog_density": 5.0,
        "wetness": 0.0,
    },
    "Fog": {
        "cloudiness": 70.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 10.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 45.0,
        "fog_density": 45.0,
        "fog_distance": 10.0,
        "fog_falloff": 1.0,
        "wetness": 20.0,
    },
}


def make_weather(name: str) -> Any:
    """Return a ``carla.WeatherParameters`` for a custom or built-in preset name."""
    import carla

    if name in WEATHER_PRESETS:
        weather = carla.WeatherParameters()
        for key, value in WEATHER_PRESETS[name].items():
            setattr(weather, key, value)
        return weather
    preset = getattr(carla.WeatherParameters, name, None)
    if not isinstance(preset, carla.WeatherParameters):
        raise ValueError(f"Unknown weather preset {name!r}")
    return preset


def needs_headlights(weather: Any) -> bool:
    """Drivers switch lights on at night, at dusk and in heavy rain or fog."""
    return (weather.sun_altitude_angle < 8.0 or weather.precipitation >= 50.0
            or weather.fog_density >= 30.0)


def weather_to_dict(weather: Any) -> Dict[str, float]:
    """Plain dict of the main weather fields (for recording metadata)."""
    keys = ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
            "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance", "wetness")
    return {k: float(getattr(weather, k)) for k in keys if hasattr(weather, k)}


@dataclass
class PoleSway:
    """Wind-driven sway of a pole-mounted camera.

    Args:
        amplitude_deg: Peak pitch/yaw sway in degrees during a full gust (0 = static camera).
            About 0.3 deg is a light breeze, 0.5-1.0 deg is strong wind on a ~11 m pole.
        pole_height: Pole height in metres; converts the bending angle into pole-top displacement.
        natural_freq_hz: First bending mode of the pole.
        gust_period_s: Typical time between gust peaks.
        seed: Random seed for phases, wind direction and jitter.
    """

    amplitude_deg: float = 0.5
    pole_height: float = 11.5
    natural_freq_hz: float = 1.1
    gust_period_s: float = 6.0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        r = self._rng
        self._phase = r.uniform(0, 2 * math.pi, size=6)
        # Main wind direction: the sway happens mostly along it, partly across it.
        self._wind_dir = r.uniform(0, 2 * math.pi)
        self._jitter_phase = r.uniform(0, 2 * math.pi, size=3)
        self._jitter_freq = r.uniform(5.0, 9.0, size=3)

    def gust(self, t: float) -> float:
        """Gust envelope in [0.35, 1.0]."""
        w = 2 * math.pi / self.gust_period_s
        g = 0.5 + 0.3 * math.sin(w * t + self._phase[4]) + 0.2 * math.sin(2.7 * w * t + self._phase[5])
        return float(np.clip(0.35 + 0.65 * g, 0.35, 1.0))

    def offsets(self, t: float) -> Tuple[float, float, float, float, float, float]:
        """Return ``(d_pitch, d_yaw, d_roll, dx, dy, dz)`` at time ``t`` (degrees, metres)."""
        if self.amplitude_deg <= 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        a = self.amplitude_deg * self.gust(t)
        f = self.natural_freq_hz
        p = self._phase
        two_pi = 2 * math.pi
        # Bending along / across the wind (first mode plus a weaker, slightly detuned second mode).
        along = math.sin(two_pi * f * t + p[0]) + 0.3 * math.sin(two_pi * 2.6 * f * t + p[1])
        across = 0.45 * math.sin(two_pi * 1.07 * f * t + p[2])
        # Small high-frequency vibration (lamp head, fixings).
        jit = [0.08 * math.sin(two_pi * jf * t + jp) for jf, jp in zip(self._jitter_freq, self._jitter_phase)]

        c, s = math.cos(self._wind_dir), math.sin(self._wind_dir)
        tilt_x = a * (along * c - across * s)  # pole tilt about the world x-axis (deg)
        tilt_y = a * (along * s + across * c)  # pole tilt about the world y-axis (deg)

        d_pitch = tilt_y + a * jit[0]
        d_yaw = 0.6 * tilt_x + a * jit[1]       # torsion and head rotation, weaker than bending
        d_roll = 0.5 * tilt_x + a * jit[2]
        h = self.pole_height
        dx = h * math.radians(tilt_y)
        dy = -h * math.radians(tilt_x)
        dz = -0.5 * h * (math.radians(tilt_x) ** 2 + math.radians(tilt_y) ** 2)
        return (d_pitch, d_yaw, d_roll, dx, dy, dz)


# ---------------------------------------------------------------------------
# Offline camera shake: re-project an already recorded frame (and its labels)
# as if the camera had rotated by the PoleSway offsets at time t.
# ---------------------------------------------------------------------------

def intrinsics(width: int, height: int, fov_deg: float = 90.0) -> np.ndarray:
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def rotation_homography(d_pitch: float, d_yaw: float, d_roll: float, K: np.ndarray) -> np.ndarray:
    """Pixel homography for a small camera rotation (degrees, CARLA pitch-up / yaw-right / roll).

    Camera frame: x right, y down, z forward. A pure rotation maps pixels exactly with
    ``H = K R^T K^-1``. The pole-top translation (centimetres at ~50 m range) is ignored.
    """
    p, y, r = (math.radians(v) for v in (d_pitch, d_yaw, d_roll))
    rx = np.array([[1, 0, 0], [0, math.cos(-p), -math.sin(-p)], [0, math.sin(-p), math.cos(-p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    R = rz @ ry @ rx
    return K @ R.T @ np.linalg.inv(K)


def warp_box(H: np.ndarray, box: Tuple[float, float, float, float], width: int, height: int,
             min_inside: float = 0.5) -> Tuple[int, int, int, int] | None:
    """Warp a pixel box with ``H``; None if less than ``min_inside`` of it stays in the image."""
    x1, y1, x2, y2 = box
    pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float64).T
    q = H @ pts
    q = q[:2] / q[2]
    nx1, ny1, nx2, ny2 = q[0].min(), q[1].min(), q[0].max(), q[1].max()
    area = max(nx2 - nx1, 0) * max(ny2 - ny1, 0)
    cx1, cy1 = max(nx1, 0), max(ny1, 0)
    cx2, cy2 = min(nx2, width - 1), min(ny2, height - 1)
    if cx2 - cx1 < 2 or cy2 - cy1 < 2 or area <= 0:
        return None
    if (cx2 - cx1) * (cy2 - cy1) < min_inside * area:
        return None
    return int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))


def motion_blur(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Linear motion blur along the pixel displacement (dx, dy) during the exposure."""
    length = math.hypot(dx, dy)
    if length < 1.0:
        return img
    n = int(math.ceil(length)) | 1
    kernel = np.zeros((n, n), np.float32)
    c = n // 2
    ux, uy = dx / length, dy / length
    for s in np.linspace(-length / 2, length / 2, max(n * 2, 3)):
        kernel[int(round(c + s * uy)), int(round(c + s * ux))] = 1.0
    kernel /= kernel.sum()
    import cv2

    return cv2.filter2D(img, -1, kernel)


def shake_frame(img: np.ndarray, boxes: list, sway: PoleSway, t: float, fov_deg: float = 90.0,
                exposure_s: float = 0.025) -> Tuple[np.ndarray, list]:
    """Apply pole sway at time ``t`` to a recorded frame and its boxes.

    ``boxes`` is a list of ``(class_id, (x1, y1, x2, y2))``; boxes pushed mostly out of the
    image are dropped. Returns the warped (and motion-blurred) image and the new boxes.
    """
    import cv2

    h, w = img.shape[:2]
    K = intrinsics(w, h, fov_deg)
    H = rotation_homography(*sway.offsets(t)[:3], K)
    out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Blur from the camera's motion during the exposure (displacement of the image centre).
    H0 = rotation_homography(*sway.offsets(t - exposure_s)[:3], K)
    c = np.array([w / 2, h / 2, 1.0])
    a, b = H @ c, H0 @ c
    out = motion_blur(out, a[0] / a[2] - b[0] / b[2], a[1] / a[2] - b[1] / b[2])
    new_boxes = []
    for cls, box in boxes:
        nb = warp_box(H, box, w, h)
        if nb is not None:
            new_boxes.append((cls, nb))
    return out, new_boxes

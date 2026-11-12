"""Registry of the roundabout maps RoundWatch can co-simulate (CARLA + SUMO).

Each map bundles everything that differs between scenarios: the SUMO config, how CARLA gets
the world (a built-in town or an OpenDRIVE file it generates a mesh from), the two camera
poses, the roundabout geometry used by the compliance rules, and the ground-plane homography.

All positions are CARLA world coordinates (metres; the y axis is mirrored w.r.t. SUMO/netedit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.perception.homography import RoadwayHomography
from src.perception.tracker import pixel_to_road_ray

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class CameraPose:
    position: Vec3   # (x, y, z) metres
    rotation: Vec3   # (pitch, yaw, roll) degrees


@dataclass(frozen=True)
class MapSpec:
    name: str
    sumo_cfg: str
    cam_south: CameraPose
    cam_north: CameraPose
    geometry: Dict[str, Any]
    carla_town: Optional[str] = None      # built-in CARLA map (e.g. "Town03_Opt")
    xodr: Optional[str] = None            # OpenDRIVE file -> client.generate_opendrive_world()
    homography_json: Optional[str] = None  # calibrated homographies; else derived from camera poses
    scenery: bool = False                  # dress the generated world (src/simulation/scenery.py)
    tracking_roi: Tuple[Tuple[float, float], float] = ((2.0, 6.0), 38.0)  # (centre, radius) where vehicles get IDs
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


MAPS: Dict[str, MapSpec] = {
    "town03": MapSpec(
        name="town03",
        description="CARLA Town03 roundabout (SUMO net converted from Town03.xodr)",
        sumo_cfg="data/sumo/roundabout.sumocfg",
        carla_town="Town03_Opt",
        cam_south=CameraPose((0.71, -26.2, 11.5), (-28.0, 88.0, 0.0)),
        cam_north=CameraPose((-2.31, 27.2, 11.5), (-28.0, -80.0, 0.0)),
        geometry={},  # RoundaboutGeometry defaults are calibrated to Town03
        homography_json="configs/homography_town03.json",
    ),
    "roundabout_2lane": MapSpec(
        name="roundabout_2lane",
        description="Synthetic 2-lane roundabout (scripts/build_roundabout_2lane.py), CARLA world generated from OpenDRIVE",
        sumo_cfg="data/sumo/roundabout_2lane/roundabout_2lane.sumocfg",
        xodr="data/sumo/roundabout_2lane/roundabout_2lane.xodr",
        # Poles on the centre line of the south / north approaches, 30 m from the centre
        cam_south=CameraPose((0.0, -30.0, 11.5), (-28.0, 90.0, 0.0)),
        cam_north=CameraPose((0.0, 30.0, 11.5), (-28.0, -90.0, 0.0)),
        geometry={"center": (0.0, 0.0), "r_island": 14.0, "r_inner": 18.0, "r_outer": 22.0},
        scenery=True,
        tracking_roi=((0.0, 0.0), 60.0),
    ),
}


def get_map(name: str) -> MapSpec:
    try:
        return MAPS[name]
    except KeyError:
        raise ValueError(f"Unknown map {name!r}; available: {', '.join(MAPS)}") from None


def homography_from_pose(cam: CameraPose, w: int = 1280, h: int = 720, fov: float = 90.0) -> RoadwayHomography:
    """Ground-plane homography (pixel <-> road) computed exactly from the camera pose."""
    corners = [(0.1 * w, 0.6 * h), (0.9 * w, 0.6 * h), (0.9 * w, 0.95 * h), (0.1 * w, 0.95 * h)]
    dst = []
    for u, v in corners:
        pt = pixel_to_road_ray(u, v, cam.position, cam.rotation, fov=fov, w=w, h=h)
        if pt is None:
            raise ValueError("Camera does not see the ground at the homography reference pixels")
        dst.append(pt)
    return RoadwayHomography(corners, dst)


def load_homographies(spec: MapSpec, w: int = 1280, h: int = 720) -> Tuple[RoadwayHomography, RoadwayHomography]:
    """(south, north) homographies: calibrated file if the map has one, else derived from the poses."""
    if spec.homography_json and Path(spec.homography_json).exists():
        with open(spec.homography_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        hs, hn = RoadwayHomography(), RoadwayHomography()
        if "cameras_720" in cfg:
            for hom, key in ((hs, "cam_south"), (hn, "cam_north")):
                hom.H = np.array(cfg["cameras_720"][key]["H_matrix"], dtype=np.float64)
                hom.H_inv = np.array(cfg["cameras_720"][key]["H_inv_matrix"], dtype=np.float64)
            return hs, hn
        hs.H = np.array(cfg["H_matrix"], dtype=np.float64)
        hs.H_inv = np.array(cfg["H_inv_matrix"], dtype=np.float64)
        return hs, hs
    return homography_from_pose(spec.cam_south, w, h), homography_from_pose(spec.cam_north, w, h)


def load_carla_world(client: Any, spec: MapSpec) -> Any:
    """Make the CARLA server run this map's world and return it."""
    import carla  # imported lazily so the registry is usable without the CARLA API

    if spec.xodr:
        xodr_text = Path(spec.xodr).read_text(encoding="utf-8")
        params = carla.OpendriveGenerationParameters(
            vertex_distance=1.0,
            max_road_length=200.0,
            wall_height=0.0,
            additional_width=0.0,  # >0 widens roads but not junction connectors -> stepped edges
            smooth_junctions=True,
            enable_mesh_visibility=True,
            enable_pedestrian_navigation=False,
        )
        return client.generate_opendrive_world(xodr_text, params)

    current = client.get_world().get_map().name
    if spec.carla_town and spec.carla_town.replace("_Opt", "") not in current:
        return client.load_world(spec.carla_town)
    return client.get_world()


def dress_world(client: Any, world: Any, spec: MapSpec) -> Optional[Any]:
    """Spawn the map's runtime scenery (buildings, markings, ...) if it has any. Returns a Scenery handle."""
    if not spec.scenery:
        return None
    from src.simulation.scenery import decorate_roundabout

    return decorate_roundabout(client, world, spec.sumo_cfg.replace(".sumocfg", ".net.xml"))

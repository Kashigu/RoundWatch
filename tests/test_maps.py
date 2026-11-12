"""Tests for the co-simulation map registry and the 2-lane roundabout assets."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.compliance.geometry import RoundaboutGeometry
from src.perception.tracker import pixel_to_road_ray
from src.simulation.maps import MAPS, get_map, homography_from_pose


def test_unknown_map_raises():
    with pytest.raises(ValueError, match="available"):
        get_map("nope")


@pytest.mark.parametrize("name", sorted(MAPS))
def test_map_assets_exist(name):
    spec = MAPS[name]
    assert Path(spec.sumo_cfg).exists()
    if spec.xodr:
        assert Path(spec.xodr).exists()


def test_pose_homography_matches_ray_projection():
    cam = get_map("roundabout_2lane").cam_south
    hom = homography_from_pose(cam)
    for u, v in [(640, 500), (300, 650), (1000, 460)]:
        exp = pixel_to_road_ray(u, v, cam.position, cam.rotation)
        got = hom.pixel_to_road(u, v)
        assert abs(got[0] - exp[0]) < 0.1 and abs(got[1] - exp[1]) < 0.1


def test_roundabout_2lane_lane_rules_in_network():
    """Entry right lane -> outer ring lane, left -> inner; exits only from the outer lane."""
    net = ET.parse(get_map("roundabout_2lane").sumo_cfg.replace(".sumocfg", ".net.xml")).getroot()
    conns = {(c.get("from"), c.get("fromLane"), c.get("to"), c.get("toLane"))
             for c in net.iter("connection") if not c.get("from").startswith(":")}
    assert ("in_E", "0", "ring_EN", "0") in conns
    assert ("in_E", "1", "ring_EN", "1") in conns
    exits_from = {fl for (f, fl, t, _tl) in conns if f.startswith("ring_") and t.startswith("out_")}
    assert exits_from == {"0"}


def test_roundabout_2lane_ring_matches_geometry_bands():
    """Ring lane centrelines fall in the inner (14-18 m) / outer (18-22 m) bands of the rules."""
    net = ET.parse(get_map("roundabout_2lane").sumo_cfg.replace(".sumocfg", ".net.xml")).getroot()
    geom = RoundaboutGeometry(**get_map("roundabout_2lane").geometry)
    for lane in net.iter("lane"):
        lid = lane.get("id")
        if not lid.startswith("ring_"):
            continue
        x, y = map(float, lane.get("shape").split()[len(lane.get("shape").split()) // 2].split(","))
        zone = geom.classify_zone(x, -y)  # CARLA frame
        assert zone == ("outer_ring" if lid.endswith("_0") else "inner_ring"), (lid, zone)


def test_scenery_markings_follow_lane_borders():
    """Road markings are offset from SUMO lane centrelines by exactly half a lane width."""
    import math

    sumolib = pytest.importorskip("sumolib")
    from src.simulation.scenery import DASH, _offset_polyline, _sample, marking_polylines

    net = sumolib.net.readNet(get_map("roundabout_2lane").sumo_cfg.replace(".sumocfg", ".net.xml"))
    lines = marking_polylines(net, (14.0, 18.0, 22.0))
    radii = {round(math.hypot(*poly[0])) for poly, _ in lines[:2]}
    assert radii == {14, 18}                       # island edge + ring lane divider
    assert lines[1][1] == DASH                     # divider is dashed
    # right border of a straight lane heading +x lies 1.75 m to its right (-y in SUMO)
    assert _offset_polyline([(0.0, 0.0), (10.0, 0.0)], 1.75)[0] == pytest.approx((0.0, -1.75))
    pts = list(_sample([(0.0, 0.0), (12.0, 0.0)], 1.0, dash=(3.0, 3.0)))
    assert len(pts) == 6 and all(p[0] < 3.0 or 6.0 <= p[0] < 9.0 for p, _ in pts)

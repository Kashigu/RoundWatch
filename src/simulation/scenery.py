"""Runtime scenery for OpenDRIVE-generated CARLA worlds (no Unreal editor needed).

CARLA builds only a bare road mesh from an .xodr file. This module dresses it at runtime with
CARLA's own packaged assets, spawned as ``static.prop.mesh`` actors:
  - grass ground under everything (it also shows through the central and splitter islands),
  - road markings drawn from the exact SUMO lane geometry (solid / dashed lines, give-way lines),
  - fountain, bushes and cypresses on the central island; street lights on the splitter islands,
  - houses, apartment blocks, trees and street lights around the roundabout.

Markings and ground are small/large flat ``SM_Plane`` tiles coloured with
``world.apply_color_texture_to_objects`` (CARLA has no non-uniform scaling or decals at runtime).
Placement is deterministic (fixed seed), so every run - and every camera dataset - looks the same.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

Pt = Tuple[float, float]

ASSETS = "/Game/Carla/Static/"
PLANE = ASSETS + "Other/SM_Plane.SM_Plane"  # 10 m x 12 m at scale 1 once laid flat
PLANE_PITCH = -90.0  # only this face takes apply_color_texture_*; the other keeps a red default material
PLANE_SIZE = (10.0, 12.0)

MARK_SCALE = 0.015            # marking tile: 0.15 m along the line x 0.18 m wide
MARK_STEP = PLANE_SIZE[0] * MARK_SCALE
MARK_Z = 0.03                 # just above the road mesh
GROUND_SCALE = 5.0            # grass tile: 50 m x 60 m
DASH = (3.0, 3.0)             # dashed line: 3 m paint, 3 m gap
PAINT_RADIUS = 52.0           # paint approach markings up to this distance from the centre
ISLAND_RADIUS = 60.0          # build raised splitter islands up to this distance
ISLAND_LINE_GAP = 0.3         # painted line runs this far outside the island curb
CURB_W = 0.36                 # curb band width (tile 0.30 m along x 0.36 m across)
CURB_SCALE = 0.03
ISLAND_Z = 0.13               # raised island surface
ISLAND_GRASS_SCALE = 0.08     # island grass tile 0.8 m x 0.96 m
GIVEWAY_SCALE = 0.04          # give-way dash tile 0.40 m along the line x 0.48 m


def _mesh_name(path: str) -> str:
    return path.split("/")[-1].split(".")[0]


BUILDINGS = [ASSETS + f"Building/{n}.{n}" for n in (
    "SM_House01", "SM_Block05", "SM_Apartment01_1", "SM_Apartment02_1", "SM_Block08", "SM_Church01",
)]
# SM_* (plain static meshes); the SMF_* foliage variants do not render when spawned as props
TREES = [ASSETS + f"Vegetation/Trees/{n}.{n}" for n in ("SM_Ash_01", "SM_Beech", "SM_Acer_02", "SM_Ash_02")]
CYPRESS = ASSETS + "Vegetation/Trees/SM_Cypress_Small.SM_Cypress_Small"
BUSHES = [ASSETS + f"Vegetation/Bushes/SM_Bush_L_v{i}.SM_Bush_L_v{i}" for i in range(1, 6)]
STREET_LIGHT = ASSETS + "Pole/StreetLights/Group1/SM_Streetlight1_group1.SM_Streetlight1_group1"
# Double-arm island light as in Town03 (BP_StreetLight_doble = these three meshes at one transform;
# its arms lie along the mesh x axis)
HIGHWAY_LIGHT = [ASSETS + f"Pole/StreetLights/Group3/{n}.{n}" for n in (
    "SM_Streetlight2_group3", "SM_Streetlight2_group3_glass_ext", "SM_Streetlight2_group3_glass_int")]
BOLLARD = ASSETS + "Pole/SM_Lights/SM_Bollard01.SM_Bollard01"


@dataclass
class Scenery:
    """Handle to everything spawned, so it can be removed again."""
    actors: List[Any] = field(default_factory=list)

    def destroy(self, client: Any) -> None:
        import carla
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in self.actors], True)
        self.actors.clear()


# ---------------------------------------------------------------- geometry helpers

def _offset_polyline(shape: Sequence[Pt], dist: float) -> List[Pt]:
    """Offset a polyline sideways (positive = to the right of the travel direction)."""
    out = []
    n = len(shape)
    for i in range(n):
        a = shape[max(i - 1, 0)]
        b = shape[min(i + 1, n - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1.0
        out.append((shape[i][0] + dist * dy / L, shape[i][1] - dist * dx / L))
    return out


def _sample(shape: Sequence[Pt], step: float, dash: Optional[Tuple[float, float]] = None,
            keep: Callable[[Pt], bool] = lambda p: True) -> Iterable[Tuple[Pt, float]]:
    """Yield (point, heading_rad) every `step` metres along a polyline (optionally dashed)."""
    s_total = 0.0
    for a, b in zip(shape, shape[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        heading = math.atan2(dy, dx)
        s = (-s_total) % step
        while s < L:
            pos = s_total + s
            if dash is None or (pos % (dash[0] + dash[1])) < dash[0]:
                p = (a[0] + dx * s / L, a[1] + dy * s / L)
                if keep(p):
                    yield p, heading
            s += step
        s_total += L


def _circle(r: float, n: int = 240) -> List[Pt]:
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n + 1)]


# ---------------------------------------------------------------- marking layout (SUMO frame)

def marking_polylines(net: Any, ring_radii: Tuple[float, float, float]) -> List[Tuple[List[Pt], Optional[Tuple[float, float]]]]:
    """(polyline, dash) pairs in SUMO coordinates for a roundabout network.

    ring_radii = (inner edge, lane divider, outer edge) of the circulatory carriageway.
    """
    r_in, r_mid, _r_out = ring_radii
    lines: List[Tuple[List[Pt], Optional[Tuple[float, float]]]] = [
        (_circle(r_in), None),         # inner edge of the ring (central island), solid
        (_circle(r_mid), DASH),        # divider between inner and outer ring lanes, dashed
    ]
    for edge in net.getEdges():
        eid = edge.getID()
        lanes = edge.getLanes()
        if eid.startswith("ring_"):
            lane0 = lanes[0]  # outer ring lane: its right border is the outer edge (broken at junctions)
            lines.append((_offset_polyline(lane0.getShape(), lane0.getWidth() / 2.0), None))
        elif eid.startswith("in_") or eid.startswith("out_"):
            for lane in lanes:
                shape = lane.getShape()
                w = lane.getWidth() / 2.0
                lines.append((_offset_polyline(shape, w), None))       # right border
                if lane.getIndex() == len(lanes) - 1:
                    lines.append((_offset_polyline(shape, -w), None))  # left border (island edge)
    return lines


def give_way_lines(net: Any, pull: float = 0.3) -> List[List[Pt]]:
    """Transverse give-way lines across each entry, perpendicular to the lanes, `pull` m before the ring."""
    out = []
    for edge in net.getEdges():
        if not edge.getID().startswith("in_"):
            continue
        lanes = edge.getLanes()
        l0 = lanes[0].getShape()
        (ax, ay), (bx, by) = l0[-2], l0[-1]
        L = math.hypot(bx - ax, by - ay) or 1.0
        ux, uy = (bx - ax) / L, (by - ay) / L
        rx, ry = uy, -ux                                  # right of travel
        px, py = bx - ux * pull, by - uy * pull
        w0 = lanes[0].getWidth() / 2.0
        span = sum(l.getWidth() for l in lanes)
        out.append([(px + rx * w0, py + ry * w0), (px + rx * (w0 - span), py + ry * (w0 - span))])
    return out


def splitter_islands(net: Any, line_gap: float = ISLAND_LINE_GAP, max_r: float = ISLAND_RADIUS) -> List[Tuple[Any, Any]]:
    """(entry edge, shapely polygon) per splitter island (SUMO frame), inset from the painted edge lines."""
    from shapely.geometry import Point, Polygon

    polys = []
    for edge in net.getEdges():
        if not edge.getID().startswith("in_"):
            continue
        out = net.getEdge("out_" + edge.getID()[3:])
        li, lo = edge.getLanes()[-1], out.getLanes()[-1]
        border_in = _offset_polyline(li.getShape(), -li.getWidth() / 2.0)    # far -> ring
        border_out = _offset_polyline(lo.getShape(), -lo.getWidth() / 2.0)   # ring -> far
        poly = Polygon(border_in + border_out).buffer(0)
        poly = poly.buffer(-line_gap).intersection(Point(0.0, 0.0).buffer(max_r, 64))
        if not poly.is_empty:
            polys.append((edge, max(getattr(poly, "geoms", [poly]), key=lambda g: g.area)))
    return polys


def _grid_tiles(poly: Any, axis_deg: float, du: float, dv: float) -> List[Tuple[Pt, float]]:
    """Centres of du x dv rectangles (aligned to axis_deg) that lie fully inside poly."""
    from shapely.geometry import Polygon

    a = math.radians(axis_deg)
    ux, uy, vx, vy = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
    minx, miny, maxx, maxy = poly.bounds
    cx0, cy0 = (minx + maxx) / 2, (miny + maxy) / 2
    half = math.hypot(maxx - minx, maxy - miny) / 2 + max(du, dv)
    out = []
    i = -half
    while i <= half:
        j = -half
        while j <= half:
            cx, cy = cx0 + i * ux + j * vx, cy0 + i * uy + j * vy
            corners = [(cx + su * du / 2 * ux + sv * dv / 2 * vx, cy + su * du / 2 * uy + sv * dv / 2 * vy)
                       for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            if poly.contains(Polygon(corners)):
                out.append(((cx, cy), a))
            j += dv
        i += du
    return out


# ---------------------------------------------------------------- spawning

class _Spawner:
    def __init__(self, client: Any, world: Any, to_carla: Callable[[Pt], Pt]) -> None:
        import carla
        self.carla = carla
        self.client, self.world, self.to_carla = client, world, to_carla
        self.bp_mesh = world.get_blueprint_library().find("static.prop.mesh")
        self.lib = world.get_blueprint_library()
        self.scenery = Scenery()

    def batch(self, items: List[Tuple[Any, Any]]) -> List[Any]:
        """Spawn (blueprint, transform) pairs; returns the spawned actor ids."""
        cmds = [self.carla.command.SpawnActor(bp, tf) for bp, tf in items]
        ids = [r.actor_id for r in self.client.apply_batch_sync(cmds, True) if not r.error]
        self.scenery.actors.extend(ids)
        return ids

    def mesh_bp(self, path: str, scale: float = 1.0) -> Any:
        bp = self.lib.find("static.prop.mesh")
        bp.set_attribute("mesh_path", path)
        bp.set_attribute("scale", f"{scale:.4f}")
        return bp

    def tf(self, x: float, y: float, z: float = 0.0, yaw: float = 0.0, pitch: float = 0.0) -> Any:
        c = self.carla
        return c.Transform(c.Location(x, y, z), c.Rotation(pitch=pitch, yaw=yaw))

    def paint(self, items: List[Tuple[Any, Any]], color: Tuple[int, int, int], noise: int = 0, seed: int = 0) -> None:
        """Spawn flat tiles and colour them (all at once, one texture)."""
        before = set(self.world.get_names_of_all_objects())
        self.batch(items)
        self.world.tick()
        names = [n for n in self.world.get_names_of_all_objects() if n not in before]
        rng = random.Random(seed)
        N = 64 if noise else 16
        tex = self.carla.TextureColor(N, N)
        for x in range(N):
            for y in range(N):
                g = rng.randint(-noise, noise) if noise else 0
                tex.set(x, y, self.carla.Color(*(max(0, min(255, c + g)) for c in color)))
        self.world.apply_color_texture_to_objects(names, self.carla.MaterialParameter.Diffuse, tex)


def decorate_roundabout(client: Any, world: Any, net_file: str, ring_radii: Tuple[float, float, float] = (14.0, 18.0, 22.0),
                        seed: int = 7) -> Scenery:
    """Dress a roundabout OpenDRIVE world. Requires SUMO's ``sumolib`` on sys.path."""
    import sumolib

    # Talk to the world through `client` (callers pass one with a long timeout): painting ~5k
    # tiles takes longer than the 2 s timeout of the CARLA-SUMO bridge's own client.
    world = client.get_world()
    net = sumolib.net.readNet(net_file)
    ox, oy = net.getLocationOffset()

    def to_carla(p: Pt) -> Pt:  # same transform as CARLA's SUMO bridge (BridgeHelper)
        return (p[0] - ox, -(p[1] - oy))

    sp = _Spawner(client, world, to_carla)
    rng = random.Random(seed)
    plane_mark = sp.mesh_bp(PLANE, MARK_SCALE)
    plane_ground = sp.mesh_bp(PLANE, GROUND_SCALE)

    # 1. Grass ground (top surface just below the road mesh)
    gw, gh = PLANE_SIZE[0] * GROUND_SCALE, PLANE_SIZE[1] * GROUND_SCALE
    extent = 140.0
    ground = []
    x = -extent
    while x < extent:
        y = -extent
        while y < extent:
            ground.append((plane_ground, sp.tf(x + gw / 2, y + gh / 2, -0.06 - 0.1 * GROUND_SCALE, pitch=PLANE_PITCH)))
            y += gh
        x += gw
    sp.paint(ground, (72, 112, 48), noise=16, seed=seed)

    # 2. Road markings
    def keep(p: Pt) -> bool:
        return math.hypot(*p) <= PAINT_RADIUS

    tiles = []
    for poly, dash in marking_polylines(net, ring_radii):
        for p, heading in _sample(poly, MARK_STEP, dash, keep):
            cx, cy = to_carla(p)
            tiles.append((plane_mark, sp.tf(cx, cy, MARK_Z, yaw=-math.degrees(heading), pitch=PLANE_PITCH)))
    plane_giveway = sp.mesh_bp(PLANE, GIVEWAY_SCALE)
    gstep = PLANE_SIZE[0] * GIVEWAY_SCALE
    for line in give_way_lines(net):
        for p, heading in _sample(line, gstep, (2 * gstep, gstep)):
            cx, cy = to_carla(p)
            tiles.append((plane_giveway, sp.tf(cx, cy, MARK_Z, yaw=-math.degrees(heading), pitch=PLANE_PITCH)))
    sp.paint(tiles, (235, 235, 230))

    # 2b. Raised splitter islands (concrete curb band + grass), Town03 style, and the central-island curb
    plane_curb = sp.mesh_bp(PLANE, CURB_SCALE)
    plane_igrass = sp.mesh_bp(PLANE, ISLAND_GRASS_SCALE)
    curb_tiles, grass_tiles = [], []
    cstep = PLANE_SIZE[0] * CURB_SCALE
    islands = splitter_islands(net)
    for edge, poly in islands:
        curb_line = poly.buffer(-CURB_W / 2.0)
        for part in getattr(curb_line, "geoms", [curb_line]):
            for p, heading in _sample(list(part.exterior.coords), cstep):
                cx, cy = to_carla(p)
                curb_tiles.append((plane_curb, sp.tf(cx, cy, ISLAND_Z + 0.005, yaw=-math.degrees(heading), pitch=PLANE_PITCH)))
        axis = math.degrees(math.atan2(*reversed(edge.getToNode().getCoord())))
        gdu, gdv = PLANE_SIZE[0] * ISLAND_GRASS_SCALE, PLANE_SIZE[1] * ISLAND_GRASS_SCALE
        for p, a in _grid_tiles(poly.buffer(-0.05), axis, gdu, gdv):
            cx, cy = to_carla(p)
            grass_tiles.append((plane_igrass, sp.tf(cx, cy, ISLAND_Z, yaw=-math.degrees(a), pitch=PLANE_PITCH)))
    for p, heading in _sample(_circle(ring_radii[0] - ISLAND_LINE_GAP - CURB_W / 2.0, 360), cstep):
        cx, cy = to_carla(p)
        curb_tiles.append((plane_curb, sp.tf(cx, cy, ISLAND_Z + 0.005, yaw=-math.degrees(heading), pitch=PLANE_PITCH)))
    sp.paint(grass_tiles, (88, 128, 56), noise=14, seed=seed + 1)
    sp.paint(curb_tiles, (196, 194, 186))

    # 3. Central island: fountain, ring of bushes, cypresses, garden lamps
    r_island = ring_radii[0]
    props = [(sp.lib.find("static.prop.fountain"), sp.tf(0.0, 0.0, 0.0))]
    for k in range(12):
        a = 2 * math.pi * (k + 0.5) / 12
        props.append((sp.mesh_bp(rng.choice(BUSHES)), sp.tf((r_island - 2.5) * math.cos(a), (r_island - 2.5) * math.sin(a), 0.0, yaw=rng.uniform(0, 360))))
    for k in range(6):
        a = 2 * math.pi * k / 6
        props.append((sp.mesh_bp(CYPRESS), sp.tf(7.5 * math.cos(a), 7.5 * math.sin(a), 0.0)))
        props.append((sp.lib.find("static.prop.gardenlamp"), sp.tf(10.0 * math.cos(a + 0.5), 10.0 * math.sin(a + 0.5), 0.0)))

    # 4. Splitter islands: Town03 double-arm highway lights, cypresses and a bollard at the nose.
    #    Lights sit behind the camera poles (30 m) so they never block the camera view.
    from shapely.geometry import LineString

    for edge, poly in islands:
        axis = math.atan2(*reversed(edge.getToNode().getCoord()))
        ux, uy = math.cos(axis), math.sin(axis)

        def on_island(r: float) -> Optional[Pt]:
            cut = LineString([(r * ux - 10 * uy, r * uy + 10 * ux), (r * ux + 10 * uy, r * uy - 10 * ux)])
            seg = cut.intersection(poly)
            return None if seg.is_empty else to_carla(tuple(seg.centroid.coords[0]))

        yaw_across = -math.degrees(axis) + 90.0   # arms span both carriageways
        for r in (34.0, 46.0):
            pt = on_island(r)
            if pt:
                for mesh in HIGHWAY_LIGHT:
                    props.append((sp.mesh_bp(mesh), sp.tf(pt[0], pt[1], ISLAND_Z, yaw=yaw_across)))
        for r in (40.0, 52.0):
            pt = on_island(r)
            if pt:
                props.append((sp.mesh_bp(CYPRESS), sp.tf(pt[0], pt[1], ISLAND_Z, yaw=rng.uniform(0, 360))))
        nose_r = min(math.hypot(*c) for c in poly.exterior.coords) + 0.8
        pt = on_island(nose_r)
        if pt:
            props.append((sp.mesh_bp(BOLLARD), sp.tf(pt[0], pt[1], ISLAND_Z, yaw=yaw_across)))

    # 5. Street lights around the ring and trees along the approaches (outside the road)
    legs = [math.degrees(math.atan2(*reversed(to_carla(e.getToNode().getCoord()))))
            for e in net.getEdges() if e.getID().startswith("in_")]
    for leg in legs:
        for off in (-45.0, 45.0):
            a = math.radians(leg + off)
            props.append((sp.mesh_bp(STREET_LIGHT), sp.tf(26.5 * math.cos(a), 26.5 * math.sin(a), 0.0,
                                                          yaw=math.degrees(a) + 180.0)))
        rad = math.radians(leg)
        ux, uy, nx, ny = math.cos(rad), math.sin(rad), -math.sin(rad), math.cos(rad)
        for d in range(40, 100, 12):
            for side in (-1, 1):
                lat = side * rng.uniform(14.5, 17.0)
                props.append((sp.mesh_bp(rng.choice(TREES), rng.uniform(0.8, 1.1)),
                              sp.tf(d * ux + lat * nx, d * uy + lat * ny, 0.0, yaw=rng.uniform(0, 360))))

    # 6. Buildings in the four quadrants between the legs, facing the roundabout
    for leg in legs:
        for k, (r, da) in enumerate(((58.0, 45.0), (82.0, 30.0), (82.0, 60.0))):
            a = math.radians(leg + da)
            props.append((sp.mesh_bp(rng.choice(BUILDINGS)), sp.tf(r * math.cos(a), r * math.sin(a), 0.0,
                                                                   yaw=math.degrees(a) + 180.0)))
        for d in (48.0, 70.0):  # bushes between the buildings
            a = math.radians(leg + 45.0)
            props.append((sp.mesh_bp(rng.choice(BUSHES)), sp.tf((d - 12) * math.cos(a) + 6, (d - 12) * math.sin(a), 0.0)))
    props.append((sp.lib.find("static.prop.busstop"), sp.tf(*to_carla((48.0, -9.5)), 0.0, yaw=0.0)))
    props.append((sp.lib.find("static.prop.bench01"), sp.tf(*to_carla((-30.0, 9.0)), 0.0, yaw=180.0)))
    sp.batch(props)

    world.tick()
    return sp.scenery

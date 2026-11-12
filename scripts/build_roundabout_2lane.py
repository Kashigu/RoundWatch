"""Build the 2-lane RoundWatch roundabout for CARLA + SUMO co-simulation.

Based on Pedro's single-lane SUMO roundabout (data/sumo/sumo.net.xml: 4 legs, ring at ~20 m),
rebuilt with a 2-lane circulatory ring and 2-lane entries so the Portuguese Highway Code
lane rules (Art. 14.º-A) can actually be exercised.

Geometry matches configs/roundabout.yaml (centre (0, 0)):
    central island r < 14 m | inner lane 14-18 m | outer lane 18-22 m
Each leg: 2-lane entry (right lane -> outer ring lane, left lane -> inner ring lane) and a
1-lane exit, 80 m long.

Outputs (data/sumo/roundabout_2lane/):
    roundabout_2lane.nod.xml / .edg.xml   plain-XML sources (editable in netedit)
    roundabout_2lane.net.xml              SUMO network
    roundabout_2lane.xodr                 OpenDRIVE map for CARLA (generate_opendrive_world)
    roundabout_2lane.rou.xml              vehicle types + labelled demand (compliant / infractions)
    roundabout_2lane.sumocfg              SUMO configuration (step 0.05 s, 20 FPS)

Coordinates: leg names follow the SUMO/netedit view (+y = north). CARLA mirrors the y axis,
so in CARLA world coordinates the "north" leg is at negative y.

Usage:
    .venv310\\Scripts\\python.exe scripts/build_roundabout_2lane.py
"""

from __future__ import annotations

import math
import random
import subprocess
from pathlib import Path

from runtime_paths import PROJECT_ROOT, sumo_binary

NAME = "roundabout_2lane"
OUT_DIR = PROJECT_ROOT / "data" / "sumo" / NAME

R_RING = 18.0        # ring edge centreline (2 lanes x 4 m -> lanes span 14-22 m)
RING_LANE_W = 4.0
APPROACH_LANE_W = 3.5
LEG_LENGTH = 80.0
SPEED_RING = 8.33    # 30 km/h
SPEED_APPROACH = 13.89  # 50 km/h

# Legs in counter-clockwise order (= circulation order for right-hand traffic)
LEGS = [("E", 0.0), ("N", 90.0), ("W", 180.0), ("S", 270.0)]

NO_CHANGE = "emergency authority"  # vClasses still allowed to cross a solid line

SIM_END = 300.0
SEED = 42


def _pt(r: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return r * math.cos(a), r * math.sin(a)


SPLITTER_W = 3.0                          # splitter-island width between entry and exit (m)
SPLITTER_PROFILE_R = (34.0, 28.0, 24.5)   # radii of the shape points nearest the ring


def _splitter_half_width(r: float) -> float:
    """Half the island width at radius r: full width out to 28 m, tapering to a nose at the ring."""
    if r >= 28.0:
        return SPLITTER_W / 2.0
    return 0.4 + (SPLITTER_W / 2.0 - 0.4) * (r - 24.5) / (28.0 - 24.5)


def _fmt(points) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_plain_xml() -> tuple[Path, Path]:
    nodes, edges = [], []
    for name, ang in LEGS:
        rx, ry = _pt(R_RING, ang)
        fx, fy = _pt(R_RING + LEG_LENGTH, ang)
        nodes.append(f'    <node id="ring_{name}" x="{rx:.2f}" y="{ry:.2f}" type="priority"/>')
        nodes.append(f'    <node id="far_{name}" x="{fx:.2f}" y="{fy:.2f}" type="dead_end"/>')
        # Splitter island: entry and exit carriageways are pushed apart by SPLITTER_W, tapering to
        # a nose near the ring. Lanes spread to the right of each edge's shape, so each shape is
        # offset towards its own lanes' side.
        side = (-math.sin(math.radians(ang)), math.cos(math.radians(ang)))  # entry lanes' side
        def leg_pts(sign: float) -> list[tuple[float, float]]:
            pts = []
            for r in (R_RING + LEG_LENGTH, *SPLITTER_PROFILE_R):
                off = sign * _splitter_half_width(r)
                bx, by = _pt(r, ang)
                pts.append((bx + off * side[0], by + off * side[1]))
            return pts
        in_shape = leg_pts(+1.0)
        out_shape = list(reversed(leg_pts(-1.0)))
        # Solid line between the two entry lanes: the lane chosen on arrival (by intended exit) is
        # kept into the ring. Only emergency/authority vehicles may cross it.
        edges.append(
            f'    <edge id="in_{name}" from="far_{name}" to="ring_{name}" numLanes="2" '
            f'speed="{SPEED_APPROACH}" width="{APPROACH_LANE_W}" priority="1" shape="{_fmt(in_shape)}">\n'
            f'        <lane index="0" changeLeft="{NO_CHANGE}"/>\n'
            f'        <lane index="1" changeRight="{NO_CHANGE}"/>\n'
            f'    </edge>'
        )
        edges.append(
            f'    <edge id="out_{name}" from="ring_{name}" to="far_{name}" numLanes="1" '
            f'speed="{SPEED_APPROACH}" width="{APPROACH_LANE_W}" priority="1" shape="{_fmt(out_shape)}"/>'
        )

    ring_ids = []
    for i, (name, ang) in enumerate(LEGS):
        nxt_name, nxt_ang = LEGS[(i + 1) % len(LEGS)]
        end_ang = nxt_ang if nxt_ang > ang else nxt_ang + 360.0
        arc = [_pt(R_RING, ang + t * (end_ang - ang) / 12.0) for t in range(13)]
        edge_id = f"ring_{name}{nxt_name}"
        ring_ids.append(edge_id)
        edges.append(
            f'    <edge id="{edge_id}" from="ring_{name}" to="ring_{nxt_name}" numLanes="2" '
            f'speed="{SPEED_RING}" width="{RING_LANE_W}" priority="2" spreadType="center" '
            f'shape="{_fmt(arc)}"/>'
        )

    ring_nodes = " ".join(f"ring_{n}" for n, _ in LEGS)
    edges.append(f'    <roundabout nodes="{ring_nodes}" edges="{" ".join(ring_ids)}"/>')

    nod = OUT_DIR / f"{NAME}.nod.xml"
    edg = OUT_DIR / f"{NAME}.edg.xml"
    nod.write_text("<nodes>\n" + "\n".join(nodes) + "\n</nodes>\n", encoding="utf-8")
    edg.write_text("<edges>\n" + "\n".join(edges) + "\n</edges>\n", encoding="utf-8")
    return nod, edg


def run_netconvert(nod: Path, edg: Path) -> None:
    netconvert = sumo_binary().parent / ("netconvert.exe" if sumo_binary().suffix == ".exe" else "netconvert")
    cmd = [
        str(netconvert),
        "--node-files", str(nod),
        "--edge-files", str(edg),
        "--output-file", str(OUT_DIR / f"{NAME}.net.xml"),
        "--opendrive-output", str(OUT_DIR / f"{NAME}.xodr"),
        "--no-turnarounds", "true",
        "--junctions.corner-detail", "8",
        "--junctions.limit-turn-speed", "5.5",
        "--rectangular-lane-cut", "true",
        "--offset.disable-normalization", "true",  # keep the roundabout centre at (0, 0)
        "--opendrive-output.straight-threshold", "1e-6",
    ]
    subprocess.run(cmd, check=True)


VTYPES = """\
    <!-- vType ids are CARLA blueprint names, so the CARLA bridge spawns that exact model. -->
    <!-- Compliant drivers: stay in the lane they entered on (keepRight 0) and change only when the route demands it. -->
    <vType id="vehicle.audi.tt" vClass="passenger" length="4.2" width="1.9" maxSpeed="13.9" accel="2.2" decel="4.0" minGap="2.5" tau="1.2" lcKeepRight="0" lcSpeedGain="0" lcStrategic="1.0" color="0,220,70"/>
    <vType id="vehicle.mercedes.coupe" vClass="passenger" length="4.7" width="2.0" maxSpeed="13.9" accel="2.2" decel="4.0" minGap="2.5" tau="1.2" lcKeepRight="0" lcSpeedGain="0" lcStrategic="1.0" color="30,140,250"/>
    <vType id="vehicle.seat.leon" vClass="passenger" length="4.3" width="1.8" maxSpeed="13.9" accel="2.2" decel="4.0" minGap="2.5" tau="1.2" lcKeepRight="0" lcSpeedGain="0" lcStrategic="1.0" color="200,200,200"/>
    <vType id="vehicle.citroen.c3" vClass="passenger" length="4.0" width="1.8" maxSpeed="13.9" accel="2.0" decel="4.0" minGap="2.5" tau="1.2" lcKeepRight="0" lcSpeedGain="0" lcStrategic="1.0" color="240,150,20"/>
    <vType id="vehicle.carlamotors.carlacola" vClass="truck" length="5.3" width="2.6" maxSpeed="11.1" accel="1.2" decel="3.0" minGap="3.0" tau="1.5" lcKeepRight="0" lcSpeedGain="0" lcStrategic="1.0" color="220,180,20"/>
    <vType id="vehicle.yamaha.yzf" vClass="motorcycle" length="2.2" width="0.8" maxSpeed="13.9" accel="3.0" decel="5.0" minGap="2.0" tau="1.0" lcKeepRight="0" lcSpeedGain="0" lcStrategic="1.0" color="230,40,230"/>
    <!-- Infraction driver: always prefers the outer (right) lane, even for exit 2+ (INFRACTION_OUTER_MISUSE). -->
    <vType id="vehicle.mini.cooper_s" vClass="passenger" length="3.8" width="1.8" maxSpeed="13.9" accel="2.2" decel="4.0" minGap="2.5" tau="1.2" lcKeepRight="10" lcSpeedGain="0" lcCooperative="0" lcStrategic="0.2" color="230,40,40"/>
"""

COMPLIANT_TYPES = ["vehicle.audi.tt", "vehicle.mercedes.coupe", "vehicle.seat.leon", "vehicle.citroen.c3",
                   "vehicle.carlamotors.carlacola", "vehicle.yamaha.yzf"]
COMPLIANT_WEIGHTS = [3, 3, 3, 3, 1, 1]
INFRACTION_TYPE = "vehicle.mini.cooper_s"


def route_edges(entry_idx: int, exit_number: int) -> list[str]:
    """Edges from leg `entry_idx` to its `exit_number`-th exit (1 = first exit, 3 = third)."""
    edges = [f"in_{LEGS[entry_idx][0]}"]
    for k in range(exit_number):
        a = LEGS[(entry_idx + k) % 4][0]
        b = LEGS[(entry_idx + k + 1) % 4][0]
        edges.append(f"ring_{a}{b}")
    edges.append(f"out_{LEGS[(entry_idx + exit_number) % 4][0]}")
    return edges


def write_routes(infraction_share: float = 0.2, mean_headway_s: float = 4.0, seed: int = SEED,
                 path: Path | None = None) -> Path:
    rng = random.Random(seed)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">', VTYPES]
    for e in range(4):
        for x in (1, 2, 3):
            lines.append(f'    <route id="r_{LEGS[e][0]}_exit{x}" edges="{" ".join(route_edges(e, x))}"/>')

    # Vehicle ids carry the ground-truth label: <entry>_exit<n>_<label>_<k>
    t, k = 0.0, 0
    while t < SIM_END - 60.0:
        t += rng.expovariate(1.0 / mean_headway_s)
        e = rng.randrange(4)
        x = rng.choice((1, 2, 3))
        if x >= 2 and rng.random() < infraction_share:
            vtype, label, lane = INFRACTION_TYPE, "INF_OUTER_MISUSE", 0  # stays outer for exit 2+
        else:
            vtype = rng.choices(COMPLIANT_TYPES, COMPLIANT_WEIGHTS)[0]
            label = "COMPLIANT_EXIT1" if x == 1 else "COMPLIANT_EXIT2PLUS"
            lane = 0 if x == 1 else 1  # exit 1 -> right entry lane, exit 2+ -> left entry lane
        k += 1
        lines.append(
            f'    <vehicle id="{LEGS[e][0]}_exit{x}_{label}_{k:03d}" type="{vtype}" route="r_{LEGS[e][0]}_exit{x}" '
            f'depart="{t:.2f}" departLane="{lane}" departSpeed="max" arrivalLane="current"/>'
        )
    lines.append("</routes>")
    path = path or OUT_DIR / f"{NAME}.rou.xml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[build] {k} vehicles written -> {path.name}")
    return path


def write_sumocfg(routes: str = f"{NAME}.rou.xml", path: Path | None = None, seed: int = SEED) -> Path:
    """SUMO config; `routes` is relative to the config's folder (alternative demands for training data)."""
    path = path or OUT_DIR / f"{NAME}.sumocfg"
    path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- RoundWatch: 2-lane roundabout for CARLA + SUMO co-simulation (generated by scripts/build_roundabout_2lane.py) -->
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{NAME}.net.xml"/>
        <route-files value="{routes}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{SIM_END:.0f}"/>
        <step-length value="0.05"/>
    </time>
    <processing>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
        <time-to-teleport value="-1"/>
    </processing>
    <random_number>
        <seed value="{seed}"/>
    </random_number>
</configuration>
""", encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nod, edg = write_plain_xml()
    run_netconvert(nod, edg)
    write_routes()
    write_sumocfg()
    print(f"[build] Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()

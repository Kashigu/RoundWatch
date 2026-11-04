import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import math
import numpy as np
from src.perception.homography import RoadwayHomography

cfg = json.load(open('configs/homography_town03.json', 'r', encoding='utf-8'))
s_cfg = cfg['cameras_720']['cam_south']
n_cfg = cfg['cameras_720']['cam_north']

h_s = RoadwayHomography()
h_n = RoadwayHomography()
h_s.H = np.array(s_cfg['H_matrix'], dtype=np.float64)
h_s.H_inv = np.array(s_cfg['H_inv_matrix'], dtype=np.float64)
h_n.H = np.array(n_cfg['H_matrix'], dtype=np.float64)
h_n.H_inv = np.array(n_cfg['H_inv_matrix'], dtype=np.float64)

gt = json.load(open('data/recordings/carla_sumo_gt.json', 'r', encoding='utf-8'))
both = [r for r in gt if r.get('bbox_south') and r.get('bbox_north')]
print(f"Total GT vehicle instances visible in both cameras: {len(both)}")

dists = []
for r in both[:20]:
    bs = r['bbox_south']
    bn = r['bbox_north']
    us, vs = (bs[0] + bs[2]) / 2.0, float(bs[3])
    # bn in gt is in full composite coordinates [1280..2560]
    un = (bn[0] + bn[2]) / 2.0 - 1280.0
    vn = float(bn[3])
    xs, ys = h_s.pixel_to_road(us, vs)
    xn, yn = h_n.pixel_to_road(un, vn)
    d = math.hypot(xs - xn, ys - yn)
    dists.append(d)
    print(f"Frame {r['frame_idx']} [{r['sumo_id']}]: South=({xs:6.2f}, {ys:6.2f}) | North=({xn:6.2f}, {yn:6.2f}) | Diff={d:.2f}m")

if dists:
    print(f"Mean distance difference between cameras on road: {np.mean(dists):.2f}m")

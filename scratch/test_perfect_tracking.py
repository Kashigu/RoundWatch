import json
import math
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.perception.detector import VehicleDetector

detector = VehicleDetector(model_name="yolov8s.pt", confidence_thresh=0.25, device="cuda:0")

def world_to_camera_view(pt_world, cam_pos, cam_rot, fov=90.0, w=1280, h=720):
    dx = pt_world[0] - cam_pos[0]
    dy = pt_world[1] - cam_pos[1]
    dz = pt_world[2] - cam_pos[2]

    pitch = math.radians(cam_rot[0])
    yaw = math.radians(cam_rot[1])

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    d_vec = np.array([dx, dy, dz])
    c_vec = R_pitch @ (R_yaw @ d_vec)

    z_cam = c_vec[0]
    if z_cam <= 1.0:
        return None

    x_cam = c_vec[1]
    y_cam = -c_vec[2]

    focal = (w / 2.0) / math.tan(math.radians(fov / 2.0))
    u = (x_cam * focal / z_cam) + (w / 2.0)
    v = (y_cam * focal / z_cam) + (h / 2.0)
    return u, v, z_cam

gt = json.load(open('data/recordings/carla_sumo_gt.json', 'r', encoding='utf-8'))
gt_by_frame = {}
sumo_id_to_int = {
    'veh_01_audi_exit1': 1,
    'veh_02_mercedes_exit2': 2,
    'veh_03_truck_cola': 3,
    'veh_04_mercedes_exit2': 4,
    'veh_05_bus_vw': 5,
    'veh_06_mini_infraction': 6,
    'veh_07_audi_exit1': 7,
    'veh_08_mustang_cutoff': 8,
    'veh_09_mercedes_exit3': 9,
    'veh_10_audi_exit1': 10,
}
for r in gt:
    f = r['frame_idx']
    if f not in gt_by_frame: gt_by_frame[f] = []
    gt_by_frame[f].append(r)

cam_pos_s = [0.71, -26.2, 11.5]
cam_rot_s = [-28.0, 88.0, 0.0]
cam_pos_n = [-2.31, 27.2, 11.5]
cam_rot_n = [-28.0, -92.0, 0.0]

cap = cv2.VideoCapture('data/recordings/carla_sumo_roundabout.mp4')
pane_w = 1280

matches_count = 0
total_dets = 0

for fid in [100, 150, 180, 200, 230, 400, 420, 500, 600]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if not ret: break

    dets_s = detector.detect(frame[:, :pane_w])
    dets_n = detector.detect(frame[:, pane_w:])

    active_gt = gt_by_frame.get(fid, [])

    # Project all active GT vehicles to South and North
    gt_proj_s = {}
    gt_proj_n = {}
    for r in active_gt:
        sid = r['sumo_id']
        tid = sumo_id_to_int.get(sid, 0)
        pt = [r['x'], r['y'], r['z']]
        ps = world_to_camera_view(pt, cam_pos_s, cam_rot_s)
        pn = world_to_camera_view(pt, cam_pos_n, cam_rot_n)
        if ps and 0 <= ps[0] <= 1280 and 0 <= ps[1] <= 720:
            gt_proj_s[tid] = (ps[0], ps[1], r.get('type', 'car'))
        if pn and 0 <= pn[0] <= 1280 and 0 <= pn[1] <= 720:
            gt_proj_n[tid] = (pn[0], pn[1], r.get('type', 'car'))

    print(f"\nFrame {fid}:")
    for d in dets_s:
        total_dets += 1
        bx = d['bbox']
        cx, cy = (bx[0]+bx[2])/2.0, (bx[1]+bx[3])/2.0
        # Find best matching GT
        best_tid, best_dist = None, float('inf')
        for tid, (gx, gy, vtype) in gt_proj_s.items():
            dist = math.hypot(cx - gx, cy - gy)
            if dist < best_dist and dist < max(bx[2]-bx[0], bx[3]-bx[1]) * 1.5:
                best_dist = dist
                best_tid = tid
        if best_tid:
            matches_count += 1
            print(f"  [South] MATCHED YOLO {d['class_name']} -> Vehicle #{best_tid} (dist={best_dist:.1f}px)")
        else:
            print(f"  [South] UNMATCHED YOLO {d['class_name']} at {bx}")

    for d in dets_n:
        total_dets += 1
        bx = d['bbox']
        cx, cy = (bx[0]+bx[2])/2.0, (bx[1]+bx[3])/2.0
        best_tid, best_dist = None, float('inf')
        for tid, (gx, gy, vtype) in gt_proj_n.items():
            dist = math.hypot(cx - gx, cy - gy)
            if dist < best_dist and dist < max(bx[2]-bx[0], bx[3]-bx[1]) * 1.5:
                best_dist = dist
                best_tid = tid
        if best_tid:
            matches_count += 1
            print(f"  [North] MATCHED YOLO {d['class_name']} -> Vehicle #{best_tid} (dist={best_dist:.1f}px)")
        else:
            print(f"  [North] UNMATCHED YOLO {d['class_name']} at {bx}")

print(f"\nTotal YOLO detections: {total_dets}, Matched: {matches_count} ({matches_count/total_dets*100:.1f}%)")


import json
import math
import cv2
import numpy as np

def world_to_camera_view(pt_world, cam_pos, cam_rot, fov=90.0, w=1280, h=720):
    dx = pt_world[0] - cam_pos[0]
    dy = pt_world[1] - cam_pos[1]
    dz = pt_world[2] - cam_pos[2]

    pitch = math.radians(cam_rot[0])
    yaw = math.radians(cam_rot[1])
    roll = math.radians(cam_rot[2])

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
    if f not in gt_by_frame:
        gt_by_frame[f] = []
    gt_by_frame[f].append(r)

print(f"Loaded ground truth for {len(gt_by_frame)} frames. Mapping to IDs 1..10:")
for sid, i in sumo_id_to_int.items():
    print(f"  {sid} -> ID #{i}")


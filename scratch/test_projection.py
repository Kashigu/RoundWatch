import json
import math
import numpy as np

def world_to_camera_view(pt_world, cam_pos, cam_rot, fov=90.0, w=1280, h=720):
    # pt_world: [x, y, z] in Unreal coordinates (X forward, Y right, Z up)
    # cam_pos: [x, y, z]
    # cam_rot: [pitch, yaw, roll] in degrees
    dx = pt_world[0] - cam_pos[0]
    dy = pt_world[1] - cam_pos[1]
    dz = pt_world[2] - cam_pos[2]

    # Convert rotations to radians
    pitch = math.radians(cam_rot[0])
    yaw = math.radians(cam_rot[1])
    roll = math.radians(cam_rot[2])

    # Rotation matrix CARLA (Unreal convention)
    # Yaw (around Z)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Pitch (around Y)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Roll (around X)
    cr, sr = math.cos(roll), math.sin(roll)

    # World to camera matrix (standard Unreal/CARLA camera coordinate: X forward, Y right, Z up)
    # Camera coordinate: X right, Y down, Z forward (OpenCV convention)
    R_yaw = np.array([
        [cy, sy, 0],
        [-sy, cy, 0],
        [0, 0, 1]
    ])
    R_pitch = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])
    d_vec = np.array([dx, dy, dz])
    c_vec = R_pitch @ (R_yaw @ d_vec)

    # In camera coordinates: c_vec[0] is forward (optical axis Z), c_vec[1] is right (X), c_vec[2] is up (-Y)
    z_cam = c_vec[0]
    if z_cam <= 0.5:
        return None  # Behind camera

    x_cam = c_vec[1]
    y_cam = -c_vec[2]

    focal = (w / 2.0) / math.tan(math.radians(fov / 2.0))
    u = (x_cam * focal / z_cam) + (w / 2.0)
    v = (y_cam * focal / z_cam) + (h / 2.0)
    return u, v, z_cam

# Test projection on frame 180
gt = json.load(open('data/recordings/carla_sumo_gt.json'))
f180 = [r for r in gt if r['frame_idx'] == 180]

cam_pos_s = [0.71, -26.2, 11.5]
cam_rot_s = [-28.0, 88.0, 0.0]
cam_pos_n = [-2.31, 27.2, 11.5]
cam_rot_n = [-28.0, -92.0, 0.0]  # looking south

print("=== 3D Projection Test Frame 180 ===")
for r in f180:
    sid = r['sumo_id']
    pt = [r['x'], r['y'], r['z']]
    proj_s = world_to_camera_view(pt, cam_pos_s, cam_rot_s)
    proj_n = world_to_camera_view(pt, cam_pos_n, cam_rot_n)
    print(f"Vehicle {sid} at {pt}:")
    if proj_s and 0 <= proj_s[0] <= 1280 and 0 <= proj_s[1] <= 720:
        print(f"  -> South: pixel=({proj_s[0]:.1f}, {proj_s[1]:.1f}), depth={proj_s[2]:.1f}m")
    if proj_n and 0 <= proj_n[0] <= 1280 and 0 <= proj_n[1] <= 720:
        print(f"  -> North: pixel=({proj_n[0]:.1f}, {proj_n[1]:.1f}), depth={proj_n[2]:.1f}m")


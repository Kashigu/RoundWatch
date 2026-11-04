import math
import numpy as np

def pixel_to_road_ray(u, v, cam_pos, cam_rot, fov=90.0, w=1280, h=720, ground_z=0.0):
    # Camera ray in camera space (X right, Y down, Z forward)
    focal = (w / 2.0) / math.tan(math.radians(fov / 2.0))
    x_cam = (u - (w / 2.0)) / focal
    y_cam = (v - (h / 2.0)) / focal
    z_cam = 1.0

    # Convert to Unreal camera space: X forward, Y right, Z up
    v_c = np.array([z_cam, x_cam, -y_cam])
    v_c /= np.linalg.norm(v_c)

    # Inverse rotation from camera to world
    pitch = math.radians(cam_rot[0])
    yaw = math.radians(cam_rot[1])

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # World to camera was R_pitch @ R_yaw
    # Camera to world is (R_pitch @ R_yaw).T = R_yaw.T @ R_pitch.T
    R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_w2c = R_pitch @ R_yaw
    R_c2w = R_w2c.T

    ray_w = R_c2w @ v_c
    # Ray origin is cam_pos
    # cam_pos + t * ray_w = [X, Y, ground_z]
    # cam_pos[2] + t * ray_w[2] = ground_z
    if abs(ray_w[2]) < 1e-4 or ray_w[2] >= 0:
        return None  # Ray pointing above horizon or parallel to ground

    t = (ground_z - cam_pos[2]) / ray_w[2]
    if t <= 0:
        return None

    pt_x = cam_pos[0] + t * ray_w[0]
    pt_y = cam_pos[1] + t * ray_w[1]
    return pt_x, pt_y

cam_pos_s = [0.71, -26.2, 11.5]
cam_rot_s = [-28.0, 88.0, 0.0]
cam_pos_n = [-2.31, 27.2, 11.5]
cam_rot_n = [-28.0, -80.0, 0.0]

# Test Frame 230:
# South car box was: (1092, 282, 1199, 341) -> foot is (1145.5, 341)
# North car box was: (30, 287, 163, 359)   -> foot is (96.5, 359)
pt_s = pixel_to_road_ray(1145.5, 341, cam_pos_s, cam_rot_s)
pt_n = pixel_to_road_ray(96.5, 359, cam_pos_n, cam_rot_n)
print("Frame 230 car:")
print("  South foot projected to road:", pt_s)
print("  North foot projected to road:", pt_n)
if pt_s and pt_n:
    print(f"  Road distance difference: {math.hypot(pt_s[0]-pt_n[0], pt_s[1]-pt_n[1]):.2f}m")

# Test Frame 200:
# South car foot: (292, 247)
# North car foot: (1216, 388)
pt_s = pixel_to_road_ray(292, 247, cam_pos_s, cam_rot_s)
pt_n = pixel_to_road_ray(1216, 388, cam_pos_n, cam_rot_n)
print("\nFrame 200 car:")
print("  South foot projected to road:", pt_s)
print("  North foot projected to road:", pt_n)
if pt_s and pt_n:
    print(f"  Road distance difference: {math.hypot(pt_s[0]-pt_n[0], pt_s[1]-pt_n[1]):.2f}m")


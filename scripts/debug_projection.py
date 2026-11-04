import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import carla
from src.simulation.camera_rig import CameraRig

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(5.0)
world = client.get_world()

rig = CameraRig(world, image_size=(1920, 1080), fov=85.0)
cam = rig.spawn_camera((-28.4, 33.0, 8.3), (-18.0, 50.0, 0.0))

sps = world.get_map().get_spawn_points()
nearby = [sp for sp in sps if 5.0 < math.hypot(sp.location.x - (-6.4), sp.location.y - 61.0) < 30.0]

bp = world.get_blueprint_library().filter("vehicle.audi.etron")[0]
veh = world.try_spawn_actor(bp, nearby[0])

world.tick()
v_loc = veh.get_location()
c_trans = cam.get_transform()
print(f"Vehicle loc: x={v_loc.x:.1f}, y={v_loc.y:.1f}, z={v_loc.z:.1f}")
print(f"Camera loc:  x={c_trans.location.x:.1f}, y={c_trans.location.y:.1f}, z={c_trans.location.z:.1f}")
print(f"Camera rot:  p={c_trans.rotation.pitch:.1f}, y={c_trans.rotation.yaw:.1f}, r={c_trans.rotation.roll:.1f}")

# Manual projection calculation
world_to_cam = np.array(c_trans.get_inverse_matrix())
p_world = np.array([v_loc.x, v_loc.y, v_loc.z, 1.0])
p_cam = np.dot(world_to_cam, p_world)
print(f"p_cam raw: {p_cam}")

# Unreal coords: X fwd, Y right, Z up
# Standard camera frame: X right, Y down, Z fwd
p_cam_std = np.array([p_cam[1], -p_cam[2], p_cam[0]])
print(f"p_cam_std (X right, Y down, Z fwd): {p_cam_std}")

p_proj = np.dot(rig.K, p_cam_std)
u = int(p_proj[0] / p_proj[2])
v = int(p_proj[1] / p_proj[2])
print(f"Projected pixel: u={u}, v={v} (Image size: 1920x1080)")
bbox = rig.compute_2d_bounding_box(veh)
print(f"Computed 2D bounding box: {bbox}")

if veh is not None:
    veh.destroy()
rig.destroy()

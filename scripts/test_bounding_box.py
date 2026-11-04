import os
import sys
import math

sys.path.insert(0, os.path.abspath("."))
import carla
from src.simulation.camera_rig import CameraRig

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(5.0)
world = client.get_world()

rig = CameraRig(world, image_size=(1920, 1080), fov=85.0)
cam = rig.spawn_camera((-28.4, 33.0, 8.3), (-45.0, 50.0, 0.0))

sps = world.get_map().get_spawn_points()
nearby = [sp for sp in sps if 5.0 < math.hypot(sp.location.x - (-6.4), sp.location.y - 61.0) < 30.0]

bp = world.get_blueprint_library().filter("vehicle.audi.etron")[0]
veh = world.try_spawn_actor(bp, nearby[0])
print("Vehicle spawned:", veh)

world.tick()
bbox = rig.compute_2d_bounding_box(veh)
print(f"Computed 2D bounding box: {bbox}")

if veh is not None:
    veh.destroy()
rig.destroy()
print("Cleaned up successfully.")


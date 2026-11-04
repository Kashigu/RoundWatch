import os
import sys
import math

sys.path.insert(0, os.path.abspath("."))
import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(5.0)
world = client.get_world()
sps = world.get_map().get_spawn_points()

cx, cy = -6.4, 61.0
print(f"Total spawn points in Town03: {len(sps)}")
nearby = []
for idx, sp in enumerate(sps):
    dist = math.hypot(sp.location.x - cx, sp.location.y - cy)
    if dist < 65.0:
        nearby.append((idx, dist, sp.location.x, sp.location.y, sp.location.z, sp.rotation.yaw))

nearby.sort(key=lambda item: item[1])
print(f"Spawn points within 65m of roundabout ({len(nearby)}):")
for idx, dist, x, y, z, yaw in nearby[:20]:
    print(f"  SP {idx:3d}: dist={dist:5.1f}m, loc=({x:6.1f}, {y:6.1f}, {z:4.1f}), yaw={yaw:5.1f}")


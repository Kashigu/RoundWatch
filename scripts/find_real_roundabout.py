import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import carla

client = carla.Client("127.0.0.1", 2000)
world = client.get_world()
carla_map = world.get_map()

# Find waypoints with roundabout junction or circular curvature
waypoints = carla_map.generate_waypoints(2.0)
print(f"Total waypoints: {len(waypoints)}")

# Look for junctions with multiple connections or roundabout tag
roundabout_candidates = []
for wp in waypoints:
    # In Town03, where is the roundabout?
    # Let's check distance to (0, 0) through (-50, 50)
    loc = wp.transform.location
    if -50 < loc.x < 50 and -50 < loc.y < 50:
        if wp.is_junction:
            j = wp.get_junction()
            roundabout_candidates.append((j.id, loc.x, loc.y, loc.z))

print(f"Candidate junction waypoints near (0,0): {len(roundabout_candidates)}")
# Group by junction ID
j_ids = {}
for j_id, x, y, z in roundabout_candidates:
    if j_id not in j_ids:
        j_ids[j_id] = []
    j_ids[j_id].append((x, y, z))

for j_id, pts in j_ids.items():
    pts_arr = np.array(pts)
    cx, cy = pts_arr[:, 0].mean(), pts_arr[:, 1].mean()
    r = np.hypot(pts_arr[:, 0] - cx, pts_arr[:, 1] - cy).max()
    print(f"Junction {j_id}: center=({cx:.1f}, {cy:.1f}), span radius={r:.1f}m, num_pts={len(pts)}")


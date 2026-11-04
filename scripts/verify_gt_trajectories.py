import json
import math

with open('data/recordings/carla_sumo_gt.json') as f:
    gt = json.load(f)

print(f"Total GT entries: {len(gt)}")

# Group by vehicle_id
by_veh = {}
for entry in gt:
    vid = entry['vehicle_id']
    if vid not in by_veh:
        by_veh[vid] = []
    by_veh[vid].append(entry)

print(f"\nUnique vehicles recorded: {len(by_veh)}")
for vid, entries in sorted(by_veh.items()):
    frames = [e['frame_idx'] for e in entries]
    print(f"Vehicle '{vid}': visible across frames {min(frames)} -> {max(frames)} ({len(entries)} detections)")

    # Check for sudden yaw flips
    flips = 0
    max_d_yaw = 0.0
    for i in range(len(entries)-1):
        # We can also check position displacement direction
        dx = entries[i+1]['x'] - entries[i]['x']
        dy = entries[i+1]['y'] - entries[i]['y']
        dt = (entries[i+1]['frame_idx'] - entries[i]['frame_idx']) / 20.0
        if dt > 0:
            speed = math.hypot(dx, dy) / dt
            # Check if moving backwards or jumping
            if speed > 25.0: # > 90 km/h jump
                print(f"  WARNING: High speed / jump on '{vid}' at frame {entries[i]['frame_idx']}: {speed:.1f} m/s")


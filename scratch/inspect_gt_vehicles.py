import json
import math

gt = json.load(open('data/recordings/carla_sumo_gt.json', 'r', encoding='utf-8'))
print(f"Total GT entries: {len(gt)}")

# Group by sumo_id
sumo_vehicles = sorted(list(set(r['sumo_id'] for r in gt)))
print("Vehicles in simulation:")
for idx, sid in enumerate(sumo_vehicles, 1):
    records = [r for r in gt if r['sumo_id'] == sid]
    f_min = min(r['frame_idx'] for r in records)
    f_max = max(r['frame_idx'] for r in records)
    vtype = records[0].get('type', 'car')
    print(f"  ID #{idx:2d} | {sid:25s} | type={vtype:25s} | frames {f_min:3d}..{f_max:3d}")


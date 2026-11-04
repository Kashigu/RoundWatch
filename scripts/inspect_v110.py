import json

with open('data/recordings/carla_sumo_gt.json') as f:
    gt = json.load(f)

v110 = [e for e in gt if e['vehicle_id'] == '110']
v110.sort(key=lambda e: e['frame_idx'])

for e in v110[50:65]:
    print(f"Frame {e['frame_idx']}: pos=({e['x']:.2f}, {e['y']:.2f}, {e['z']:.2f}) bbox={e['bbox']}")


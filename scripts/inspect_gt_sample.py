import json

with open('data/recordings/carla_sumo_gt.json') as f:
    gt = json.load(f)

print(gt[0])
v_ids = set(e['vehicle_id'] for e in gt)
print("Unique vehicle IDs in GT:", v_ids)

target_id = list(v_ids)[0]
v_entries = [e for e in gt if e['vehicle_id'] == target_id]
v_entries.sort(key=lambda e: e['frame_idx'])
for e in v_entries[50:65]:
    print(f"Frame {e['frame_idx']}: pos=({e['x']:.2f}, {e['y']:.2f}, {e['z']:.2f})")


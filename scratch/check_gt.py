import json

gt = json.load(open('data/recordings/carla_sumo_gt.json', 'r', encoding='utf-8'))
vehicles = {}
for r in gt:
    sid = r.get('sumo_id')
    if sid not in vehicles:
        vehicles[sid] = {'type': r.get('type'), 'frames': []}
    vehicles[sid]['frames'].append(r['frame_idx'])

print(f"Total vehicles in ground truth: {len(vehicles)}")
for sid, v in vehicles.items():
    print(f"  {sid}: type={v['type']}, frames {min(v['frames'])}..{max(v['frames'])} ({len(v['frames'])} frames)")


import json

gt = json.load(open('data/recordings/carla_sumo_gt.json', 'r', encoding='utf-8'))
v1_south = [r for r in gt if r['sumo_id'] == 'veh_01_audi_exit1' and r.get('bbox_south')]
v1_north = [r for r in gt if r['sumo_id'] == 'veh_01_audi_exit1' and r.get('bbox_north')]

print(f"veh_01 visible in South: {len(v1_south)} frames ({min(r['frame_idx'] for r in v1_south)}..{max(r['frame_idx'] for r in v1_south)})")
print(f"veh_01 visible in North: {len(v1_north)} frames ({min(r['frame_idx'] for r in v1_north)}..{max(r['frame_idx'] for r in v1_north)})")


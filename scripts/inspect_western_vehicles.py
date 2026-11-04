import json
import math

with open('data/recordings/carla_sumo_gt.json') as f:
    gt = json.load(f)

for target_sumo in ['veh_02_mercedes_exit2', 'veh_03_truck_cola', 'veh_05_bus_vw', 'veh_09_mercedes_exit3']:
    v_entries = [e for e in gt if e['sumo_id'] == target_sumo]
    v_entries.sort(key=lambda e: e['frame_idx'])
    print(f"\n==================== {target_sumo} ({len(v_entries)} frames) ====================")
    
    # Filter to western quadrant: around x ~ 0 to -35, y ~ 0 to 20 (or CARLA roundabout coords)
    # Let's inspect step-by-step positions and displacement angles
    prev_e = None
    flips = []
    for e in v_entries:
        if prev_e is not None:
            dx = e['x'] - prev_e['x']
            dy = e['y'] - prev_e['y']
            dist = math.hypot(dx, dy)
            df = e['frame_idx'] - prev_e['frame_idx']
            if dist > 0.05:
                ang = math.degrees(math.atan2(dy, dx))
                e['move_ang'] = ang
                if 'move_ang' in prev_e:
                    d_ang = abs(ang - prev_e['move_ang'])
                    if d_ang > 180:
                        d_ang = 360 - d_ang
                    if d_ang > 60.0:
                        flips.append((prev_e['frame_idx'], e['frame_idx'], prev_e['move_ang'], ang, d_ang, (e['x'], e['y'])))
        prev_e = e

    if flips:
        print(f"WARNING: {len(flips)} sudden directional reversals found!")
        for fl in flips:
            print(f"  Frame {fl[0]}->{fl[1]}: angle jump {fl[4]:.1f} deg (from {fl[2]:.1f} to {fl[3]:.1f}) at pos {fl[5]}")
    else:
        print("PERFECT: ZERO directional reversals! Trajectory advances continuously and smoothly!")


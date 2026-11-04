import math

def check_angles(name, pts_str):
    pts = [list(map(float, p.split(','))) for p in pts_str.split()]
    print(f"\n=== Angles for {name} ===")
    angles = []
    for i in range(len(pts)-1):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        # In SUMO, angle is in degrees, 0 is North (dy > 0), 90 is East (dx > 0), 180 is South (dy < 0), 270 is West (dx < 0)
        # SUMO angle = (math.atan2(dx, dy) * 180 / math.pi) % 360
        sumo_ang = (math.atan2(dx, dy) * 180.0 / math.pi) % 360.0
        carla_yaw = (math.atan2(dy, dx) * 180.0 / math.pi)
        angles.append((sumo_ang, carla_yaw))
        print(f"Seg {i}: ({pts[i][0]},{pts[i][1]}) -> ({pts[i+1][0]},{pts[i+1][1]}) | SUMO ang: {sumo_ang:.1f} deg | CARLA yaw: {carla_yaw:.1f} deg")
    
    # Check max delta between consecutive segments
    for i in range(len(angles)-1):
        diff = abs(angles[i+1][0] - angles[i][0])
        if diff > 180:
            diff = 360 - diff
        if diff > 45:
            print(f"WARNING: Large angular change of {diff:.1f} deg between seg {i} and {i+1}!")

shape_3_0 = "153.30,221.94 143.18,220.12 135.79,215.12 131.15,206.93 129.24,195.56 129.40,193.63 130.46,190.19 133.22,185.83"
shape_3_1 = "153.37,218.44 144.70,216.85 138.37,212.49 134.38,205.34 132.74,195.42 132.77,194.83 133.63,191.77 136.03,187.91"
shape_1_0 = "145.59,228.46 142.41,220.56 136.97,212.06 131.74,203.53 129.24,195.56 129.40,193.63 130.46,190.19 133.22,185.83"
shape_1_1 = "149.04,227.90 145.88,220.07 140.44,211.70 135.23,203.32 132.74,195.42 132.77,194.83 133.63,191.77 136.03,187.91"

check_angles("3_0 (-37 to -13 lane 0)", shape_3_0)
check_angles("3_1 (-37 to -13 lane 1)", shape_3_1)
check_angles("1_0 (-25 to -13 lane 0)", shape_1_0)
check_angles("1_1 (-25 to -13 lane 1)", shape_1_1)


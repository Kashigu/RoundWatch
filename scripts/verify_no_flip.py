import os
import sys

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"E:\Simulation_Engines\SUMO"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

sumo_binary = r"E:\Simulation_Engines\SUMO\bin\sumo.exe"
sumo_cmd = [sumo_binary, "-c", "data/sumo/roundabout.sumocfg", "--no-step-log"]

traci.start(sumo_cmd)

prev_angles = {}
max_delta_per_veh = {}
jump_detected = []

step = 0
while traci.simulation.getMinExpectedNumber() > 0 and step < 1200: # up to 60s
    traci.simulationStep()
    step += 1
    
    veh_ids = traci.vehicle.getIDList()
    for vid in veh_ids:
        ang = traci.vehicle.getAngle(vid) # 0..360
        pos = traci.vehicle.getPosition(vid)
        lane = traci.vehicle.getLaneID(vid)
        
        if vid in prev_angles:
            delta = abs(ang - prev_angles[vid])
            if delta > 180:
                delta = 360 - delta
            
            # Record max delta per step (each step is 0.05s)
            # Normal turning at 8 m/s on a roundabout (radius ~ 25m):
            # angular velocity ~ v/R = 8 / 25 = 0.32 rad/s = 18.3 deg/s.
            # In 0.05s step, delta should be ~ 0.9 deg.
            # A 180-deg flip would have delta > 100 deg in 1 step!
            if vid not in max_delta_per_veh or delta > max_delta_per_veh[vid]:
                max_delta_per_veh[vid] = delta
            
            if delta > 45.0:
                jump_detected.append({
                    "step": step,
                    "time": step * 0.05,
                    "veh": vid,
                    "delta": delta,
                    "old_ang": prev_angles[vid],
                    "new_ang": ang,
                    "lane": lane,
                    "pos": pos
                })
        prev_angles[vid] = ang

traci.close()

print(f"Simulation completed ({step} steps).")
print("\nMax angle change in a single 0.05s step per vehicle:")
for vid, md in sorted(max_delta_per_veh.items()):
    print(f"  {vid}: {md:.2f} deg")

if jump_detected:
    print(f"\nWARNING: {len(jump_detected)} yaw jumps detected!")
    for j in jump_detected:
        print(f"  At t={j['time']:.2f}s: veh '{j['veh']}' jumped {j['delta']:.1f} deg (from {j['old_ang']:.1f} to {j['new_ang']:.1f}) on lane {j['lane']}")
else:
    print("\nSUCCESS: 0 yaw jumps detected across all vehicles! Traversal is completely smooth!")


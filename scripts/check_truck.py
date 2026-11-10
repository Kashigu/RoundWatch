import os
import sys

from runtime_paths import configured_path, sumo_binary

sumo_home = configured_path("SUMO_HOME")
os.environ["SUMO_HOME"] = str(sumo_home)
sys.path.append(str(sumo_home / "tools"))
import traci

sumo_cmd = [str(sumo_binary()), "-c", "data/sumo/roundabout.sumocfg", "--no-step-log"]

traci.start(sumo_cmd)

step = 0
prev_ang = None
while traci.simulation.getMinExpectedNumber() > 0 and step < 1200:
    traci.simulationStep()
    step += 1
    if "veh_03_truck_cola" in traci.vehicle.getIDList():
        ang = traci.vehicle.getAngle("veh_03_truck_cola")
        pos = traci.vehicle.getPosition("veh_03_truck_cola")
        lane = traci.vehicle.getLaneID("veh_03_truck_cola")
        speed = traci.vehicle.getSpeed("veh_03_truck_cola")
        if prev_ang is not None:
            delta = abs(ang - prev_ang)
            if delta > 180:
                delta = 360 - delta
            if delta > 10:
                print(f"t={step*0.05:.2f}s: delta={delta:.1f} deg, ang={ang:.1f}, lane={lane}, pos={pos}, speed={speed:.1f}")
        prev_ang = ang

traci.close()

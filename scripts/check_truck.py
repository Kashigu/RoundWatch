import os
import sys

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = r"E:\Simulation_Engines\SUMO"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

sumo_binary = r"E:\Simulation_Engines\SUMO\bin\sumo.exe"
sumo_cmd = [sumo_binary, "-c", "data/sumo/roundabout.sumocfg", "--no-step-log"]

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


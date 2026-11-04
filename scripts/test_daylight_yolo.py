import os
import sys
import cv2

sys.path.insert(0, os.path.abspath("."))
import carla
from src.perception.detector import VehicleDetector
from src.simulation.camera_rig import CameraRig

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(5.0)
world = client.get_world()

# Set bright clear noon weather
world.set_weather(carla.WeatherParameters.ClearNoon)
print("Set weather to ClearNoon.")

rig = CameraRig(world, image_size=(1920, 1080), fov=85.0)
cam = rig.spawn_camera((-28.4, 33.0, 8.3), (-18.0, 50.0, 0.0))

sps = world.get_map().get_spawn_points()
nearby = [sp for sp in sps if 5.0 < ( (sp.location.x - (-6.4))**2 + (sp.location.y - 61.0)**2 )**0.5 < 25.0]

bp = world.get_blueprint_library().filter("vehicle.audi.etron")[0]
veh = world.try_spawn_actor(bp, nearby[0])

# Tick simulation to render
for _ in range(5):
    world.tick()

frame = rig.latest_frame
if frame is not None:
    print("Frame captured. Mean BGR:", frame.mean(axis=(0,1)))
    cv2.imwrite("data/recordings/test_clearnoon_frame.png", frame)
    detector = VehicleDetector(model_name="yolov8n.pt", confidence_thresh=0.20, device="cuda:0")
    dets = detector.detect(frame, target_classes=[2, 3, 5, 7])
    print(f"YOLO Detections count: {len(dets)}")
    for d in dets:
        print("  Detected:", d)

if veh is not None:
    veh.destroy()
rig.destroy()


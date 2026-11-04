import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import carla
from ultralytics import YOLO

client = carla.Client("127.0.0.1", 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1920")
cam_bp.set_attribute("image_size_y", "1080")
cam_bp.set_attribute("fov", "85")

# Test camera from Central Island looking out vs perimeter looking in
# Center is (-6.4, 61.0, 0.3)
tests = [
    ("perimeter_south", (-6.4, 61.0 - 26.0, 8.0), (-25.0, 90.0, 0.0)),
    ("perimeter_west",  (-6.4 - 26.0, 61.0, 8.0), (-25.0, 0.0, 0.0)),
    ("perimeter_north", (-6.4, 61.0 + 26.0, 8.0), (-25.0, -90.0, 0.0)),
    ("central_island",  (-6.4, 61.0, 8.0),        (-25.0, 45.0, 0.0)),
]

model = YOLO("yolov8n.pt")

for name, pos, rot in tests:
    trans = carla.Transform(carla.Location(*pos), carla.Rotation(*rot))
    cam = world.spawn_actor(cam_bp, trans)
    img_holder = []
    cam.listen(lambda img: img_holder.append(img))
    for _ in range(5):
        world.tick()
    cam.stop()
    cam.destroy()

    if img_holder:
        img_data = img_holder[-1]
        raw = np.frombuffer(img_data.raw_data, dtype=np.uint8)
        bgr = raw.reshape((1080, 1920, 4))[:, :, :3]
        out_path = f"data/recordings/view_{name}.png"
        cv2.imwrite(out_path, bgr)
        print(f"Saved {out_path} | Mean BGR: {bgr.mean(axis=(0,1))}")
        preds = model.predict(source=bgr, conf=0.25, verbose=False)[0]
        print(f"  YOLO Detections in {name}: {len(preds.boxes)}")

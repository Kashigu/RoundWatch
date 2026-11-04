import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import carla

client = carla.Client("127.0.0.1", 2000)
world = client.get_world()

# Center of roundabout is around (2.0, 6.0, 0.3)
cx, cy, cz = 2.0, 6.0, 0.3

bp_lib = world.get_blueprint_library()
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1920")
cam_bp.set_attribute("image_size_y", "1080")
cam_bp.set_attribute("fov", "90")

# Bird's eye view from top down: 45m up
trans_top = carla.Transform(
    carla.Location(x=cx, y=cy, z=cz + 45.0),
    carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
)
cam = world.spawn_actor(cam_bp, trans_top)
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
    cv2.imwrite("data/recordings/real_roundabout_topdown.png", bgr)
    print("Saved real_roundabout_topdown.png!")


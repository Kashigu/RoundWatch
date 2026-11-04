import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
import carla

client = carla.Client("127.0.0.1", 2000)
world = client.get_world()

cx, cy, cz = 2.0, 6.0, 0.3
cam_pos = (cx, cy - 38.0, cz + 16.0)
cam_rot = (-26.0, 90.0, 0.0)

bp_lib = world.get_blueprint_library()
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1920")
cam_bp.set_attribute("image_size_y", "1080")
cam_bp.set_attribute("fov", "85")

trans = carla.Transform(
    carla.Location(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2]),
    carla.Rotation(pitch=cam_rot[0], yaw=cam_rot[1], roll=cam_rot[2])
)

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
    cv2.imwrite("data/recordings/carla_roundabout_elevated_view.png", bgr)
    print("Saved carla_roundabout_elevated_view.png!")


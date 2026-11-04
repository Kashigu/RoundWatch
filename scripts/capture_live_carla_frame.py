"""Capture a live photorealistic frame of Town03 Roundabout from CARLA 0.9.15."""

from pathlib import Path
import time
import numpy as np
import cv2
import carla


def capture_roundabout_frame(output_image_path: str = "data/recordings/carla_town03_roundabout_live.png"):
    print("Connecting to live CARLA instance...")
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Ensure Town03
    if "Town03" not in world.get_map().name:
        print("Switching world to Town03...")
        world = client.load_world("Town03")
        time.sleep(2.0)

    # Find the roundabout location
    # In Town03, the 5-way multi-lane roundabout is centered around X=0, Y=50 or nearby
    # Let's inspect spawn points near X=0
    sp_list = world.get_map().get_spawn_points()
    roundabout_sp = [sp for sp in sp_list if abs(sp.location.x) < 30.0 and abs(sp.location.y) < 70.0]
    
    if roundabout_sp:
        target_loc = roundabout_sp[0].location
    else:
        target_loc = carla.Location(x=0.0, y=0.0, z=0.0)

    print(f"Target Roundabout Focus Area: X={target_loc.x:.1f}, Y={target_loc.y:.1f}, Z={target_loc.z:.1f}")

    # Camera mounting transform: elevated 18m above, pitch -50 deg pointing down
    cam_transform = carla.Transform(
        carla.Location(x=target_loc.x - 20.0, y=target_loc.y - 25.0, z=target_loc.z + 18.0),
        carla.Rotation(pitch=-45.0, yaw=45.0, roll=0.0)
    )

    bp_library = world.get_blueprint_library()
    cam_bp = bp_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1920")
    cam_bp.set_attribute("image_size_y", "1080")
    cam_bp.set_attribute("fov", "90")

    camera_actor = world.spawn_actor(cam_bp, cam_transform)
    print(f"Spawned virtual camera actor ID: {camera_actor.id}")

    captured_frame = None

    def on_image(image):
        nonlocal captured_frame
        # Convert raw CARLA BGRA buffer to BGR numpy array
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = raw.reshape((image.height, image.width, 4))
        captured_frame = bgra[:, :, :3].copy()

    camera_actor.listen(on_image)

    # Wait up to 5 seconds for a frame
    print("Waiting for camera sensor frame...")
    start_time = time.time()
    while captured_frame is None and (time.time() - start_time) < 6.0:
        world.wait_for_tick()

    camera_actor.stop()
    camera_actor.destroy()
    print("Cleaned up camera sensor actor.")

    if captured_frame is not None:
        out_path = Path(output_image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), captured_frame)
        print(f"[SUCCESS] Saved live photorealistic CARLA frame to {out_path}")

        # Copy to artifact directory for display
        art_path = Path(r"C:\Users\orlan\.gemini\antigravity\brain\dc5e9231-1ac9-4f55-b832-205cbb1301e5\carla_live_town03.png")
        cv2.imwrite(str(art_path), captured_frame)
        print(f"Saved artifact preview to {art_path}")
        return True
    else:
        print("[ERROR] Failed to capture frame from camera sensor!")
        return False


if __name__ == "__main__":
    capture_roundabout_frame()


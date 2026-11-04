import os
import sys
import traceback

sys.path.insert(0, os.path.abspath("."))
with open("cleanup_log.txt", "w") as f:
    try:
        import carla
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        f.write(f"Connected to world: {world.get_map().name}\n")
        f.flush()

        from src.simulation.camera_rig import CameraRig
        rig = CameraRig(world)
        cam = rig.spawn_camera((-28.4, 33.0, 8.3), (-45.0, 50.0, 0.0))
        f.write("Camera spawned.\n")
        f.flush()

        world.tick()
        f.write(f"Tick completed. Latest frame shape: {rig.latest_frame.shape if rig.latest_frame is not None else None}\n")
        f.flush()

        f.write("Attempting rig.destroy()...\n")
        f.flush()
        rig.destroy()
        f.write("rig.destroy() succeeded!\n")
        f.flush()

    except Exception as e:
        f.write(f"EXCEPTION: {e}\n")
        traceback.print_exc(file=f)


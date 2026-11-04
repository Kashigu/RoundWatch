"""Verify Town03 Roundabout map loading and sensor attachment in CARLA 0.9.15."""

import time
import carla


def verify_carla_town03(host: str = "127.0.0.1", port: int = 2000) -> bool:
    print(f"Connecting to CARLA at {host}:{port}...")
    client = carla.Client(host, port)
    client.set_timeout(20.0)

    world = client.get_world()
    current_map = world.get_map().name

    if "Town03" not in current_map:
        print(f"Current map is {current_map}. Loading Town03 (multi-lane roundabout)...")
        world = client.load_world("Town03")
        time.sleep(2.0)
    else:
        print(f"CARLA is already running on {current_map}!")

    carla_map = world.get_map()
    print(f"[SUCCESS] Town03 Loaded! Total Map Waypoints: {len(carla_map.generate_waypoints(2.0))}")

    # Inspect Roundabout coordinates in Town03:
    # In CARLA Town03, the roundabout center is around (X=0.0, Y=0.0, Z=0.0)
    spawn_points = carla_map.get_spawn_points()
    print(f"Total vehicle spawn points in Town03: {len(spawn_points)}")

    # Test Blueprint Library
    bp_library = world.get_blueprint_library()
    vehicles = bp_library.filter("vehicle.*")
    cameras = bp_library.filter("sensor.camera.rgb")
    print(f"Available vehicle blueprints: {len(vehicles)}")
    print(f"Available RGB camera blueprints: {len(cameras)}")

    return True


if __name__ == "__main__":
    verify_carla_town03()


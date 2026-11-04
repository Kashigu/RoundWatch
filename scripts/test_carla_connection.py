"""Test connection to running CARLA simulator instance."""

import sys
import carla


def test_connection(host: str = "127.0.0.1", port: int = 2000, timeout: float = 5.0) -> bool:
    print(f"Attempting to connect to CARLA simulator at {host}:{port} (timeout: {timeout}s)...")
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        map_name = world.get_map().name
        print(f"[SUCCESS] Connected to CARLA!")
        print(f"Current World Map: {map_name}")
        return True
    except RuntimeError as e:
        print(f"[OFFLINE] Could not connect to CARLA server: {e}")
        print("Note: Launch CarlaUE4.exe first before running live simulation.")
        return False


if __name__ == "__main__":
    test_connection()


import os
import sys
import traceback

sys.path.insert(0, os.path.abspath("."))

with open("debug_log.txt", "w") as f:
    try:
        f.write("Starting test...\n")
        f.flush()
        import carla
        f.write(f"CARLA module imported: {carla}\n")
        f.flush()
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        f.write(f"Connected to CARLA world: {world.get_map().name}\n")
        f.flush()
        from src.simulation.scenario_runner import CarlaScenarioRunner
        runner = CarlaScenarioRunner(camera_height=8.0)
        f.write("Runner instantiated.\n")
        f.flush()
        runner.connect()
        f.write("Runner connected.\n")
        f.flush()
        runner.setup_camera()
        f.write("Camera setup.\n")
        f.flush()
        runner.spawn_traffic(num_vehicles=5)
        f.write(f"Traffic spawned: {len(runner.spawned_vehicles)} vehicles.\n")
        f.flush()
        runner.run_recording(duration_sec=15.0)
        f.write("Recording completed successfully.\n")
        f.flush()
        runner.cleanup()
        f.write("Cleaned up successfully.\n")
    except Exception as e:
        f.write(f"EXCEPTION: {e}\n")
        traceback.print_exc(file=f)

import psutil

for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmd = " ".join(p.info['cmdline'] or [])
        if "debug_carla_launch.py" in cmd:
            print(f"Terminating lingering debug process PID {p.info['pid']}...")
            p.terminate()
            p.wait(timeout=2)
            print(f"Terminated PID {p.info['pid']}")
    except Exception as e:
        print(f"Error terminating: {e}")


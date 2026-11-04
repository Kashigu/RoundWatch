import psutil

print("=== CURRENTLY RUNNING PROCESSES (CARLA, PYTHON, SUMO) ===")
found = 0
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
    try:
        name = p.info['name'].lower()
        if any(k in name for k in ['carla', 'python', 'sumo']):
            cmd = " ".join(p.info['cmdline'] or [])
            mem_mb = (p.info['memory_info'].rss) / (1024 * 1024) if p.info.get('memory_info') else 0
            print(f"PID {p.info['pid']:5d} | {p.info['name']:25s} | Mem: {mem_mb:7.1f} MB | Cmd: {cmd[:100]}")
            found += 1
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if not found:
    print("No CARLA, SUMO, or Python processes found running.")
else:
    print(f"\nTotal processes found: {found}")


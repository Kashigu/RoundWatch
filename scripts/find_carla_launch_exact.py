import json

from runtime_paths import configured_path

with open(configured_path("TRANSCRIPT_FILE"), "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        idx = obj.get("step_index")
        if idx in range(350, 460) or idx in range(600, 640):
            calls = obj.get("tool_calls", [])
            for c in calls:
                cmd = c.get("args", {}).get("CommandLine", "")
                if "Carla" in cmd or "carla" in cmd:
                    print(f"Step {idx}: {cmd}")

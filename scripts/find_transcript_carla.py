import json

from runtime_paths import configured_path

with open(configured_path("TRANSCRIPT_FILE"), "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        calls = obj.get("tool_calls", [])
        for c in calls:
            cmd = c.get("args", {}).get("CommandLine", "")
            if "CarlaUE4" in cmd or "start_carla" in cmd or "2000" in cmd:
                print(f"Step {obj.get('step_index')}: {cmd[:150]}")

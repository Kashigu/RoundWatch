import json

with open(r"C:\Users\orlan\.gemini\antigravity\brain\dc5e9231-1ac9-4f55-b832-205cbb1301e5\.system_generated\logs\transcript.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        calls = obj.get("tool_calls", [])
        for c in calls:
            cmd = c.get("args", {}).get("CommandLine", "")
            if "CarlaUE4" in cmd or "start_carla" in cmd or "2000" in cmd:
                print(f"Step {obj.get('step_index')}: {cmd[:150]}")


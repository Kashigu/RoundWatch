import json

from runtime_paths import configured_path

with open(configured_path("TRANSCRIPT_FILE"), "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        idx = obj.get("step_index")
        if 1480 <= idx <= 1520:
            calls = obj.get("tool_calls", [])
            for c in calls:
                print(f"Step {idx}: {c.get('name')} {c.get('args')}")

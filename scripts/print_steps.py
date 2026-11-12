import json

from runtime_paths import configured_path

with open(configured_path("TRANSCRIPT_FILE"), "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("step_index") in [1453, 1476, 1480]:
            print(f"=== Step {obj.get('step_index')} ===")
            for c in obj.get("tool_calls", []):
                print(c.get("name"), c.get("args"))

import json

from runtime_paths import configured_path

with open(configured_path("TRANSCRIPT_FILE"), "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        idx = obj.get("step_index")
        if 1300 <= idx <= 1335:
            if obj.get("type") == "USER_INPUT":
                print(f"User at {idx}: {obj.get('content')}")
            for c in obj.get("tool_calls", []):
                print(f"Tool at {idx}: {c.get('name')} {c.get('args').get('CommandLine')}")

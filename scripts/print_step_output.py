import json

from runtime_paths import configured_path

with open(configured_path("TRANSCRIPT_FILE"), "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("step_index") in [1481, 1482, 1483, 1484, 1485]:
            print(f"Step {obj.get('step_index')}: {obj.get('type')}")
            content = obj.get('content')
            if content:
                print(content[:300])

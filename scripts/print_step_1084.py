import json

from runtime_paths import configured_path

with open(configured_path("TRANSCRIPT_FILE"), "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("step_index") == 1084:
            print("Step 1084 command:")
            for c in obj.get("tool_calls", []):
                print(c.get("args", {}).get("CommandLine"))

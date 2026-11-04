import json

with open(r"C:\Users\orlan\.gemini\antigravity\brain\dc5e9231-1ac9-4f55-b832-205cbb1301e5\.system_generated\logs\transcript.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("step_index") in [1453, 1476, 1480]:
            print(f"=== Step {obj.get('step_index')} ===")
            for c in obj.get("tool_calls", []):
                print(c.get("name"), c.get("args"))


import json

with open(r"C:\Users\orlan\.gemini\antigravity\brain\dc5e9231-1ac9-4f55-b832-205cbb1301e5\.system_generated\logs\transcript.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        idx = obj.get("step_index")
        if 1480 <= idx <= 1520:
            calls = obj.get("tool_calls", [])
            for c in calls:
                print(f"Step {idx}: {c.get('name')} {c.get('args')}")


import json

REGISTRY_PATH = "data/registry.json"

with open(REGISTRY_PATH) as f:
    reg = json.load(f)

print(f"Before: {len(reg)} entries")

# Deduplicate by college+program key, keep last occurrence
seen = {}
for entry in reg:
    key = (entry.get("college", ""), entry.get("program", ""))
    seen[key] = entry

cleaned = list(seen.values())

with open(REGISTRY_PATH, "w") as f:
    json.dump(cleaned, f, indent=2)

print(f"After: {len(cleaned)} entries")
print("Done. Registry cleaned.")

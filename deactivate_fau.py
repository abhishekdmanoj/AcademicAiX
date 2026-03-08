import json

REGISTRY_PATH = "data/registry.json"

with open(REGISTRY_PATH) as f:
    reg = json.load(f)

count = 0
for entry in reg:
    if entry.get("college") == "Florida Atlantic University":
        entry["is_active"] = False
        count += 1

with open(REGISTRY_PATH, "w") as f:
    json.dump(reg, f, indent=2)

print(f"Deactivated {count} FAU programs.")
print(f"Total registry entries: {len(reg)}")

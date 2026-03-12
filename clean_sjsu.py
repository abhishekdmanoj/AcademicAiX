import json

path = "data/registry.json"

with open(path, "r") as f:
    registry = json.load(f)

cleaned = [e for e in registry if e.get("college") != "San Jose State University"]

print(f"Removed {len(registry) - len(cleaned)} SJSU entries")
print(f"Remaining: {len(cleaned)} entries")

with open(path, "w") as f:
    json.dump(cleaned, f, indent=2)
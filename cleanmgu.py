import json

registry_path = r"C:\study\projects\Main_project\AcademicAiX\data\registry.json"

with open(registry_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Keep only entries that are NOT MGU
filtered = [entry for entry in data if entry["college"] != "Mahatma Gandhi University"]

removed = len(data) - len(filtered)

with open(registry_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2)

print(f"Removed {removed} Mahatma Gandhi University entries from registry.json")
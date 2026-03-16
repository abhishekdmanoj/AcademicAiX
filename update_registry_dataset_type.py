import json
import os

REGISTRY_PATH = os.path.join("data", "registry.json")

def update_registry():
    if not os.path.exists(REGISTRY_PATH):
        print("registry.json not found")
        return

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    updated_count = 0

    for entry in registry:
        college = entry.get("college", "").lower()

        if "san jose state university" in college:
            entry["dataset_type"] = "course_scrape"
        else:
            entry["dataset_type"] = "program"

        updated_count += 1

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4)

    print(f"Updated {updated_count} registry entries.")
    print("dataset_type field added successfully.")

if __name__ == "__main__":
    update_registry()
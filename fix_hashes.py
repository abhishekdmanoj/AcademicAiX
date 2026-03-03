import hashlib
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


with open(REGISTRY_PATH, "r") as f:
    registry = json.load(f)

for entry in registry:
    if not entry.get("hash"):
        file_path = entry["file_path"]
        if os.path.exists(file_path):
            print(f"Computing hash for {entry['program']}...")
            entry["hash"] = sha256(file_path)
            print(f"  Done: {entry['hash'][:16]}...")
        else:
            print(f"⚠ File not found: {file_path}")

with open(REGISTRY_PATH, "w") as f:
    json.dump(registry, f, indent=2)

print("\n✅ Registry hashes updated.")

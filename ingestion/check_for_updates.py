import os
import json
import hashlib
import requests
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")
TEMP_DOWNLOAD = os.path.join(PROJECT_ROOT, "temp_latest.pdf")


def compute_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_pdf(url, save_path):
    response = requests.get(url, stream=True, timeout=15)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)


def main():
    print("🔎 Checking for syllabus updates...\n")

    if not os.path.exists(REGISTRY_PATH):
        print("❌ registry.json not found.")
        return

    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    updated = False

    for entry in registry:
        if not entry.get("is_active", False):
            continue

        program = entry["program"]
        source_url = entry.get("source_url")

        if not source_url:
            print(f"⚠ No source_url for {program}")
            continue
        if not source_url.endswith(".pdf"):
            print(f"⚠ {program} source_url is not a direct PDF, skipping.")
            continue

        print(f"📄 Checking {program}...")

        try:
            download_pdf(source_url, TEMP_DOWNLOAD)
            new_hash = compute_sha256(TEMP_DOWNLOAD)

            if new_hash != entry["hash"]:
                print("🚨 Update detected!")
                entry["is_active"] = False
                new_entry = entry.copy()
                new_entry["hash"] = new_hash
                new_entry["is_active"] = True
                new_entry["academic_year"] = "UPDATED"
                new_entry["last_checked"] = str(datetime.now().date())
                registry.append(new_entry)
                updated = True
            else:
                print("✅ No changes detected.")
                entry["last_checked"] = str(datetime.now().date())

            os.remove(TEMP_DOWNLOAD)

        except Exception as e:
            print(f"❌ Error checking {program}: {e}")

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    if updated:
        print("\n♻ Updates detected. Rebuilding FAISS index...")
        try:
            from offline_pipeline.build_syllabus_index import build_syllabus_index
            build_syllabus_index()
            print("✅ Index rebuilt successfully.")
        except Exception as e:
            print(f"❌ Index rebuild failed: {e}")
            print("   Run manually: python -m offline_pipeline.build_syllabus_index")
    else:
        print("\n✔ All syllabi are up to date.")


if __name__ == "__main__":
    main()
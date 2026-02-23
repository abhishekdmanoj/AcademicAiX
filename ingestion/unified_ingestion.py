import os
import json
import hashlib
import requests
from datetime import datetime

from offline_pipeline.build_syllabus_index import build_index


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")
RAW_PDF_PATH = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")


# ---------------------------------
# TRUSTED SOURCES (PROGRAM-LEVEL)
# ---------------------------------
TRUSTED_SOURCES = [
    {
        "college": "IIT Delhi",
        "program": "M.Tech Chemical Engineering",
        "pdf_url": "https://example.com/iit_delhi_syllabus.pdf",
        "academic_year": "2025-2026"
    }
]


# ---------------------------------
# Utility: SHA-256 Hash
# ---------------------------------
def compute_sha256(file_path):
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------
# Load / Save Registry
# ---------------------------------
def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return []

    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)


# ---------------------------------
# Download PDF
# ---------------------------------
def download_pdf(url, save_path):
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Failed to download: {url}")
        return False

    with open(save_path, "wb") as f:
        f.write(response.content)

    return True


# ---------------------------------
# Ingestion Logic (Program-Level)
# ---------------------------------
def ingest_program(config):
    college = config["college"]
    program = config["program"]
    pdf_url = config["pdf_url"]
    academic_year = config.get("academic_year", "Unknown")

    print(f"\n🔍 Checking updates for {college} - {program}...")

    os.makedirs(RAW_PDF_PATH, exist_ok=True)

    safe_name = f"{college}_{program}".replace(" ", "_").lower()
    temp_file_path = os.path.join(RAW_PDF_PATH, f"{safe_name}_temp.pdf")

    if not download_pdf(pdf_url, temp_file_path):
        return False

    new_hash = compute_sha256(temp_file_path)

    registry = load_registry()

    # Check if identical version already exists
    for entry in registry:
        if (
            entry.get("college") == college and
            entry.get("program") == program and
            entry.get("hash") == new_hash
        ):
            print("✅ No changes detected (hash match).")
            os.remove(temp_file_path)
            return False

    # Mark old versions inactive
    for entry in registry:
        if (
            entry.get("college") == college and
            entry.get("program") == program
        ):
            entry["is_active"] = False

    # Rename file to permanent versioned name
    final_filename = f"{safe_name}_{academic_year}.pdf"
    final_path = os.path.join(RAW_PDF_PATH, final_filename)

    os.rename(temp_file_path, final_path)

    new_entry = {
        "college": college,
        "program": program,
        "file_path": f"data/raw_pdfs/{final_filename}",
        "hash": new_hash,
        "academic_year": academic_year,
        "is_active": True,
        "last_checked": datetime.now().strftime("%Y-%m-%d")
    }

    registry.append(new_entry)
    save_registry(registry)

    print("🆕 New version ingested.")

    return True


# ---------------------------------
# Main Unified Ingestion Runner
# ---------------------------------
def run_ingestion():
    print("🚀 Starting unified ingestion engine...")

    updated = False

    for config in TRUSTED_SOURCES:
        changed = ingest_program(config)
        if changed:
            updated = True

    if updated:
        print("🔄 Rebuilding FAISS index...")
        build_index()
        print("✅ Index rebuild complete.")
    else:
        print("ℹ No updates found. Index unchanged.")


if __name__ == "__main__":
    run_ingestion()
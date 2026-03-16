import json
import os

REGISTRY_PATH = "data/registry.json"
RAW_PDFS_PATH = "data/raw_pdfs"

WHOLE_WORD_JUNK = [
    "na", "n/a", "not provided", "degree type and subject",
    "degree type", "n_a", "not_provided"
]

FILENAME_JUNK_KEYWORDS = [
    "gender", "harassment", "medal", "award", "web-links", "web_links",
    "constitution", "fellowship", "tribal", "sexual", "yoga",
    "hostel", "tender", "notice", "circular", "refund", "withdrawal",
    "verification", "cgpa", "pgp", "sfs", "brochure",
    "ugc_approval", "ugc-approval",
    "institutional_development", "idp_of",
    "marksheet", "admit_card", "hall_ticket"
]

PROGRAM_JUNK_KEYWORDS = [
    "institutional development plan",
    "ugc approval",
    "sexual harassment",
    "gender sensitization",
]

def is_junk(entry):
    program = entry.get("program", "").strip()
    file_path = entry.get("file_path", "").lower()
    filename = os.path.basename(file_path).lower()
    program_lower = program.lower().strip()

    if program_lower in WHOLE_WORD_JUNK:
        return True, f"placeholder program name: '{program}'"
    for kw in PROGRAM_JUNK_KEYWORDS:
        if kw in program_lower:
            return True, f"junk program keyword: '{kw}'"
    for kw in FILENAME_JUNK_KEYWORDS:
        if kw in filename:
            return True, f"junk filename keyword: '{kw}'"
    return False, ""

with open(REGISTRY_PATH) as f:
    registry = json.load(f)

print(f"Registry entries before: {len(registry)}")

clean_registry = []
removed = []

for entry in registry:
    junk, reason = is_junk(entry)
    if junk:
        removed.append((entry, reason))
    else:
        clean_registry.append(entry)

print(f"Entries to remove: {len(removed)}")
print()

deleted_files = 0
missing_files = 0

for entry, reason in removed:
    file_path = entry.get("file_path", "")
    print(f"  Removing: {entry.get('college')} — {entry.get('program')}")
    print(f"  Reason  : {reason}")

    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        print(f"  Deleted : {file_path}")
        deleted_files += 1
    else:
        filename = os.path.basename(file_path)
        alt_path = os.path.join(RAW_PDFS_PATH, filename)
        if os.path.exists(alt_path):
            os.remove(alt_path)
            print(f"  Deleted : {alt_path}")
            deleted_files += 1
        else:
            print(f"  Missing : {file_path} (already gone)")
            missing_files += 1
    print()

with open(REGISTRY_PATH, "w") as f:
    json.dump(clean_registry, f, indent=2)

print("=" * 70)
print(f"Registry entries after : {len(clean_registry)}")
print(f"Entries removed        : {len(removed)}")
print(f"PDF files deleted      : {deleted_files}")
print(f"Files already missing  : {missing_files}")
print()
print("Done. Run 'python offline_pipeline/build_syllabus_index.py' to rebuild index.")

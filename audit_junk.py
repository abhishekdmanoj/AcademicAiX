import json
import os
import re

REGISTRY_PATH = "data/registry.json"

# These must match as whole words only (not substrings)
WHOLE_WORD_JUNK = [
    "na", "n/a", "not provided", "degree type and subject",
    "degree type", "n_a", "not_provided"
]

# These are substring matches on filename only (not program name)
FILENAME_JUNK_KEYWORDS = [
    "gender", "harassment", "medal", "award", "web-links", "web_links",
    "constitution", "fellowship", "tribal", "sexual", "yoga",
    "hostel", "tender", "notice", "circular", "refund", "withdrawal",
    "verification", "cgpa", "pgp", "sfs", "brochure",
    "ugc_approval", "ugc-approval",
    "institutional_development", "idp_of",
    "marksheet", "admit_card", "hall_ticket"
]

# These match anywhere in program name (specific enough to be safe)
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

    # Whole word match on program name
    program_lower = program.lower().strip()
    if program_lower in WHOLE_WORD_JUNK:
        return True, f"placeholder program name: '{program}'"

    # Specific program name substring matches
    for kw in PROGRAM_JUNK_KEYWORDS:
        if kw in program_lower:
            return True, f"junk program keyword: '{kw}'"

    # Filename substring matches
    for kw in FILENAME_JUNK_KEYWORDS:
        if kw in filename:
            return True, f"junk filename keyword: '{kw}'"

    return False, ""

with open(REGISTRY_PATH) as f:
    registry = json.load(f)

print(f"Total registry entries: {len(registry)}")
print("=" * 70)

junk_entries = []
for entry in registry:
    junk, reason = is_junk(entry)
    if junk:
        junk_entries.append((entry, reason))

print(f"\nJunk entries found: {len(junk_entries)}")
print("-" * 70)
for entry, reason in junk_entries:
    print(f"  College : {entry.get('college')}")
    print(f"  Program : {entry.get('program')}")
    print(f"  File    : {os.path.basename(entry.get('file_path', ''))}")
    print(f"  Reason  : {reason}")
    print(f"  Active  : {entry.get('is_active')}")
    print()

print("=" * 70)
print("Run cleanup_junk.py to delete these entries and their PDF files.")
print("Review the list above carefully before running cleanup.")
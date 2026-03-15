import json
import os

REGISTRY_PATH = "data/registry.json"
METADATA_PATH = "data/university_metadata.json"


def get_admission_fallback(country, degree_level):
    if country and country.strip().lower() == "india":
        return [{
            "name": "Check University Website",
            "website": "",
            "note": "Admission requirements not automatically detected. Please check the university website for entrance exam details."
        }]
    elif degree_level and degree_level.upper() == "PG":
        return [{
            "name": "No Entrance Exam",
            "website": "",
            "note": "Admission is typically based on undergraduate GPA, transcripts, and letters of recommendation. Check the university website for specific requirements."
        }]
    else:
        return [{
            "name": "No Entrance Exam",
            "website": "",
            "note": "Admission is typically based on high school grades or predicted grades. Check the university website for specific entry requirements."
        }]


with open(REGISTRY_PATH) as f:
    registry = json.load(f)

with open(METADATA_PATH) as f:
    metadata = json.load(f)

# Build a lookup from (college, program) -> registry entry for country + degree_level
registry_lookup = {}
for entry in registry:
    key = (entry.get("college"), entry.get("program"))
    registry_lookup[key] = entry

updated = 0
skipped = 0

for m in metadata:
    college = m.get("college")
    program = m.get("program")
    existing_exams = m.get("entrance_exams", [])

    # Only backfill if entrance_exams is empty
    if existing_exams:
        skipped += 1
        continue

    # Look up country and degree_level from registry
    reg_entry = registry_lookup.get((college, program))
    if not reg_entry:
        print(f"  ⚠️  No registry entry found for: {college} — {program}, skipping")
        skipped += 1
        continue

    country = reg_entry.get("country", "")
    degree_level = reg_entry.get("degree_level", "UG")

    fallback = get_admission_fallback(country, degree_level)
    m["entrance_exams"] = fallback

    print(f"  ✅ Backfilled: {college} — {program}")
    print(f"     Country: {country} | Degree: {degree_level}")
    print(f"     Note: {fallback[0]['note'][:80]}...")
    print()
    updated += 1

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print("=" * 70)
print(f"Updated  : {updated}")
print(f"Skipped  : {skipped} (already had entrance exam data)")
print(f"Total    : {len(metadata)}")
print()
print("Done. No index rebuild needed.")

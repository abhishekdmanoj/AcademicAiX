import json

registry = json.load(open('data/registry.json'))
valid = {(e['college'], e['program']) for e in registry}

meta = json.load(open('data/university_metadata.json'))
cleaned = [e for e in meta if (e['college'], e['program']) in valid]

json.dump(cleaned, open('data/university_metadata.json', 'w'), indent=2)
print(f'Kept {len(cleaned)} entries, removed {len(meta) - len(cleaned)}')

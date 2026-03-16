import json
path = 'data/registry.json'
r = json.load(open(path))
active = [e for e in r if e.get('is_active')]
print(f'Removing {len(r) - len(active)} inactive entries')
json.dump(active, open(path, 'w'), indent=2)

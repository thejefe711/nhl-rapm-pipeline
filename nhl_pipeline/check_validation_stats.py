import json
with open('data/on_ice_validation.json') as f:
    v = json.load(f)
print(f"Total games: {len(v)}")
print(f"Passed games: {len([x for x in v if x.get('all_passed')])}")
s2425 = [x for x in v if x.get('season') == '20242025']
print(f"20242025 total games: {len(s2425)}")
print(f"20242025 passed games: {len([x for x in s2425 if x.get('all_passed')])}")
if s2425:
    print("\nSample 20242025 validation:")
    print(json.dumps(s2425[0], indent=2))

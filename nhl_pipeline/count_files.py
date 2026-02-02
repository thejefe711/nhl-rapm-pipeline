from pathlib import Path

s = Path('staging/20242025')
c = Path('canonical/20242025')

print(f'Staging shifts: {len(list(s.glob("*_shifts.parquet")))}')
print(f'Canonical on_ice: {len(list(c.glob("*_event_on_ice.parquet")))}')

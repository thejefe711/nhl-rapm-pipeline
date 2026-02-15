import os
from pathlib import Path

def count_staging_games():
    staging = Path('staging')
    if not staging.exists():
        print("Staging directory not found.")
        return
    
    seasons = sorted([d for d in staging.iterdir() if d.is_dir()])
    print("=== Staging Game Counts (Shifts) ===")
    for s in seasons:
        games = list(s.glob('*_shifts.parquet'))
        print(f"{s.name}: {len(games)} games")

if __name__ == "__main__":
    count_staging_games()

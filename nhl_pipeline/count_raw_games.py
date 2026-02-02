import os
from pathlib import Path

def count_raw_games():
    raw = Path('raw')
    if not raw.exists():
        print("Raw directory not found.")
        return
    
    seasons = sorted([d for d in raw.iterdir() if d.is_dir()])
    print("=== Raw Game Counts ===")
    for s in seasons:
        games = [g for g in s.iterdir() if g.is_dir()]
        print(f"{s.name}: {len(games)} games")

if __name__ == "__main__":
    count_raw_games()

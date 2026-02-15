#!/usr/bin/env python3
"""Batch fetch player positions from NHL API using threading."""

import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_DIR = Path(__file__).parent / "profile_data"
CACHE_FILE = DATA_DIR / "position_cache.json"

def fetch_position(player_id):
    """Fetch single player position."""
    try:
        url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return player_id, data.get("position", "F")
        return player_id, "F"
    except Exception:
        return player_id, "F"


def main():
    import duckdb
    
    # Get all player IDs
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    player_ids = con.execute("SELECT DISTINCT player_id FROM apm_results").df()["player_id"].tolist()
    con.close()
    
    print(f"Total players: {len(player_ids)}")
    
    # Load existing cache
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        cache = {int(k): v for k, v in cache.items()}
    else:
        cache = {}
    
    # Find missing
    missing = [pid for pid in player_ids if pid not in cache]
    print(f"Missing positions: {len(missing)}")
    
    if not missing:
        print("All positions cached!")
        return
    
    # Fetch in parallel
    print("Fetching positions with 10 threads...")
    results = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_position, pid): pid for pid in missing}
        
        for i, future in enumerate(as_completed(futures)):
            player_id, pos = future.result()
            results[player_id] = pos
            cache[player_id] = pos
            
            if (i + 1) % 100 == 0:
                print(f"  Fetched {i + 1}/{len(missing)}")
    
    # Save cache
    cache_data = {str(k): v for k, v in cache.items()}
    CACHE_FILE.write_text(json.dumps(cache_data), encoding="utf-8")
    
    # Stats
    positions = list(cache.values())
    print(f"\nPosition distribution:")
    for pos in set(positions):
        count = positions.count(pos)
        print(f"  {pos}: {count}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Retry failed games from the fetch progress file."""
import json
import time
import random
from pathlib import Path
import requests

WEB_API = "https://api-web.nhle.com/v1"
STATS_API = "https://api.nhle.com/stats/rest/en"

def fetch_json(url: str, retries: int = 3, delay: float = 2.0):
    for attempt in range(retries):
        try:
            time.sleep(delay + random.uniform(0.5, 1.5))
            resp = requests.get(url, timeout=30)
            if resp.status_code == 429:
                wait = delay * (2 ** attempt) + random.uniform(1, 5)
                print(f"    Rate limited, waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None

def fetch_game(game_id: int, output_dir: Path) -> bool:
    season = str(game_id)[:4] + str(int(str(game_id)[:4]) + 1)
    game_dir = output_dir / season / str(game_id)
    game_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch play-by-play
    pbp = fetch_json(f"{WEB_API}/gamecenter/{game_id}/play-by-play")
    if pbp is None:
        return False
    with open(game_dir / "play_by_play.json", "w") as f:
        json.dump(pbp, f)
    
    # Fetch boxscore
    box = fetch_json(f"{WEB_API}/gamecenter/{game_id}/boxscore")
    if box is None:
        return False
    with open(game_dir / "boxscore.json", "w") as f:
        json.dump(box, f)
    
    # Fetch shifts
    shifts = fetch_json(f"{STATS_API}/shiftcharts?cayenneExp=gameId={game_id}")
    if shifts is None:
        return False
    if isinstance(shifts.get("data"), list) and len(shifts["data"]) < 100:
        return False
    with open(game_dir / "shifts.json", "w") as f:
        json.dump(shifts, f)
    
    return True

def main():
    progress_path = Path(__file__).parent / "data" / "fetch_progress.json"
    output_dir = Path(__file__).parent / "raw"
    
    with open(progress_path) as f:
        progress = json.load(f)
    
    all_failed = []
    for season, data in progress.items():
        for gid in data.get("failed_games", []):
            all_failed.append((season, gid))
    
    print(f"Found {len(all_failed)} failed games to retry")
    
    success = 0
    still_failed = []
    
    for i, (season, gid) in enumerate(all_failed, 1):
        print(f"[{i}/{len(all_failed)}] {season} Game {gid}...", end=" ")
        if fetch_game(gid, output_dir):
            print("OK")
            success += 1
            # Update progress file
            progress[season]["failed_games"].remove(gid)
            progress[season]["fetched_games"] += 1
        else:
            print("FAIL")
            still_failed.append(gid)
    
    # Save updated progress
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    
    print(f"\nDone: {success}/{len(all_failed)} recovered, {len(still_failed)} still failed")
    if still_failed:
        print(f"Still failed: {still_failed[:10]}{'...' if len(still_failed) > 10 else ''}")

if __name__ == "__main__":
    main()

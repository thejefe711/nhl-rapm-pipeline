#!/usr/bin/env python3
"""Test why certain games fail to fetch shift data."""

import requests
import json
from pathlib import Path

# Load failed games from progress
progress = json.load(open(Path(__file__).parent / "data" / "fetch_progress.json"))
failed_games = progress.get("20252026", {}).get("failed_games", [])[:10]

print("Testing NHL API for failed games:")
print("=" * 60)

for game_id in failed_games:
    # Test shift API
    shift_url = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
    try:
        r = requests.get(shift_url, timeout=30)
        data = r.json()
        shift_count = len(data.get("data", []))
        total = data.get("total", 0)
        print(f"Game {game_id}: {shift_count} shifts (total={total}, status={r.status_code})")
    except Exception as e:
        print(f"Game {game_id}: ERROR - {e}")

print("\n\nTesting a known good game:")
good_game = 2025020002
shift_url = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={good_game}"
r = requests.get(shift_url, timeout=30)
data = r.json()
print(f"Game {good_game}: {len(data.get('data', []))} shifts")

# Check if failed games have play-by-play data
print("\n\nChecking if failed games have play-by-play:")
for game_id in failed_games[:3]:
    pbp_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    try:
        r = requests.get(pbp_url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            plays = len(data.get("plays", []))
            state = data.get("gameState", "UNKNOWN")
            print(f"Game {game_id}: {plays} plays, state={state}")
        else:
            print(f"Game {game_id}: HTTP {r.status_code}")
    except Exception as e:
        print(f"Game {game_id}: ERROR - {e}")

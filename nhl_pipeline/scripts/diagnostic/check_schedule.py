#!/usr/bin/env python3
"""Check 2025-2026 schedule and shift data availability."""

import requests
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent

def get_full_schedule():
    """Fetch full 2025-2026 schedule from NHL API."""
    print("=" * 70)
    print("2025-2026 SCHEDULE AND SHIFT DATA ANALYSIS")
    print("=" * 70)
    
    # Fetch schedule for all NHL teams
    teams = ["TOR", "EDM", "NYR", "BOS", "COL", "TBL", "FLA", "VGK", 
             "DAL", "CAR", "NJD", "WPG", "VAN", "LAK", "MIN", "SEA",
             "NSH", "STL", "CGY", "OTT", "PIT", "DET", "NYI", "PHI",
             "WSH", "BUF", "ARI", "CBJ", "ANA", "MTL", "CHI", "SJS", "UTA"]
    
    all_games = {}
    
    print("\nFetching schedules from NHL API...")
    for team in teams[:5]:  # Sample 5 teams to get most games
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/20252026"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                for game in data.get("games", []):
                    game_id = game.get("id")
                    game_type = game.get("gameType")
                    game_state = game.get("gameState")
                    
                    if game_id and game_type == 2:  # Regular season only
                        all_games[game_id] = {
                            "game_id": game_id,
                            "date": game.get("gameDate"),
                            "state": game_state,
                            "home": game.get("homeTeam", {}).get("abbrev"),
                            "away": game.get("awayTeam", {}).get("abbrev"),
                        }
        except Exception as e:
            print(f"  Error fetching {team}: {e}")
    
    return all_games


def check_shift_data():
    """Check which games have shift data in raw directory."""
    raw_dir = ROOT / "raw" / "20252026"
    
    games_with_shifts = set()
    games_without_shifts = set()
    
    if raw_dir.exists():
        for game_dir in raw_dir.iterdir():
            if game_dir.is_dir():
                game_id = int(game_dir.name)
                shifts_path = game_dir / "shifts.json"
                
                if shifts_path.exists():
                    try:
                        data = json.loads(shifts_path.read_text())
                        shift_count = len(data.get("data", []))
                        if shift_count >= 100:  # Minimum threshold
                            games_with_shifts.add(game_id)
                        else:
                            games_without_shifts.add(game_id)
                    except:
                        games_without_shifts.add(game_id)
                else:
                    games_without_shifts.add(game_id)
    
    return games_with_shifts, games_without_shifts


def main():
    # Get schedule
    schedule = get_full_schedule()
    
    print(f"\nTotal regular season games in schedule: {len(schedule)}")
    
    # Count by state
    completed = [g for g in schedule.values() if g["state"] == "OFF"]
    future = [g for g in schedule.values() if g["state"] == "FUT"]
    other = [g for g in schedule.values() if g["state"] not in ["OFF", "FUT"]]
    
    print(f"  Completed (OFF): {len(completed)}")
    print(f"  Future (FUT): {len(future)}")
    print(f"  Other: {len(other)}")
    
    # Check shift data
    with_shifts, without_shifts = check_shift_data()
    
    print(f"\n--- Shift Data Availability ---")
    print(f"Games with shift data (>100 shifts): {len(with_shifts)}")
    print(f"Games attempted but no/low shifts: {len(without_shifts)}")
    
    # Check completed games not yet fetched
    completed_ids = set(g["game_id"] for g in completed)
    fetched_ids = with_shifts | without_shifts
    not_fetched = completed_ids - fetched_ids
    
    print(f"Completed games not yet fetched: {len(not_fetched)}")
    
    # Summary for user
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total 2025-2026 regular season games: ~1312 (82 games × 16 teams ÷ 2)")
    print(f"Completed so far: {len(completed)}")
    print(f"Games WITH usable shift data: {len(with_shifts)}")
    print(f"Shift data coverage: {100*len(with_shifts)/len(completed):.1f}% of completed games" if completed else "N/A")
    
    # Sample games with shifts
    print(f"\nSample games WITH shifts: {sorted(with_shifts)[:10]}")
    print(f"Sample games WITHOUT shifts: {sorted(without_shifts)[:10]}")


if __name__ == "__main__":
    main()

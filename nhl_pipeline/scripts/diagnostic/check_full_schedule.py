#!/usr/bin/env python3
"""Check full 2025-2026 schedule for all completed games."""

import requests
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent

def get_full_schedule():
    """Fetch full 2025-2026 schedule by querying the league schedule."""
    print("=" * 70)
    print("FULL 2025-2026 SCHEDULE CHECK")
    print("=" * 70)
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Use the schedule API to get all regular season games
    all_games = {}
    
    # Query by date range for the 2025-2026 season (Oct 2025 - April 2026)
    # Regular season typically runs Oct to April
    start_date = "2025-10-01"
    end_date = "2026-04-30"
    
    print(f"\nFetching schedule from {start_date} to {end_date}...")
    
    url = f"https://api-web.nhle.com/v1/schedule/{start_date}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            # Get the regularSeasonStartDate and regularSeasonEndDate
            reg_start = data.get("regularSeasonStartDate")
            reg_end = data.get("regularSeasonEndDate")
            print(f"Season dates: {reg_start} to {reg_end}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Scan through each week from October 2025
    from datetime import timedelta
    current = datetime(2025, 10, 1)
    end = datetime.now()  # Only check up to today
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                for week in data.get("gameWeek", []):
                    for game in week.get("games", []):
                        game_id = game.get("id")
                        game_type = game.get("gameType")
                        game_state = game.get("gameState")
                        game_date = week.get("date")
                        
                        # Only regular season (type=2) and completed (OFF)
                        if game_id and game_type == 2:
                            all_games[game_id] = {
                                "game_id": game_id,
                                "date": game_date,
                                "state": game_state,
                            }
        except Exception as e:
            pass  # Skip failed requests
        
        # Move forward by 7 days
        current += timedelta(days=7)
    
    return all_games

def check_local_data():
    """Check which games we have locally."""
    raw_dir = ROOT / "raw" / "20252026"
    
    games_fetched = set()
    games_with_shifts = set()
    
    if raw_dir.exists():
        for game_dir in raw_dir.iterdir():
            if game_dir.is_dir():
                game_id = int(game_dir.name)
                games_fetched.add(game_id)
                
                shifts_path = game_dir / "shifts.json"
                if shifts_path.exists():
                    try:
                        data = json.loads(shifts_path.read_text())
                        shift_count = len(data.get("data", []))
                        if shift_count >= 100:
                            games_with_shifts.add(game_id)
                    except:
                        pass
    
    return games_fetched, games_with_shifts

def main():
    # Get full schedule
    schedule = get_full_schedule()
    
    # Get completed games
    completed = {gid: g for gid, g in schedule.items() if g["state"] == "OFF"}
    
    print(f"\n--- SCHEDULE SUMMARY ---")
    print(f"Total regular season games found in schedule: {len(schedule)}")
    print(f"Completed games (state=OFF): {len(completed)}")
    
    # Check local data
    fetched, with_shifts = check_local_data()
    
    print(f"\n--- LOCAL DATA ---")
    print(f"Games in raw/20252026: {len(fetched)}")
    print(f"Games with usable shifts (>100): {len(with_shifts)}")
    
    # Find gaps
    completed_ids = set(completed.keys())
    not_fetched = completed_ids - fetched
    
    print(f"\n--- GAPS ---")
    print(f"Completed games NOT fetched: {len(not_fetched)}")
    
    if not_fetched:
        print(f"\nSample games not fetched: {sorted(not_fetched)[:20]}")
    
    # Coverage
    if completed_ids:
        coverage = 100 * len(with_shifts) / len(completed_ids)
        print(f"\n--- COVERAGE ---")
        print(f"Shift data coverage: {len(with_shifts)}/{len(completed_ids)} = {coverage:.1f}%")

if __name__ == "__main__":
    main()

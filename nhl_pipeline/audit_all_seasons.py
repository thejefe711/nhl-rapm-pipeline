#!/usr/bin/env python3
"""Audit all seasons to check for missing games based on full league schedule."""

import requests
import json
from pathlib import Path
from datetime import datetime
import time

ROOT = Path(__file__).parent
WEB_API = "https://api-web.nhle.com/v1"

TEAMS = [
    "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL",
    "CBJ", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
    "NSH", "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS",
    "SEA", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"
]

def get_full_season_schedule(season):
    """Fetch all games for a season by querying all teams."""
    all_games = {}
    print(f"  Fetching full schedule for {season}...")
    for team in TEAMS:
        url = f"{WEB_API}/club-schedule-season/{team}/{season}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                for game in data.get("games", []):
                    game_id = game.get("id")
                    game_type = game.get("gameType")
                    # Only regular season (2)
                    if game_id and game_type == 2:
                        all_games[game_id] = game
        except Exception as e:
            print(f"    Error fetching {team} for {season}: {e}")
    return all_games

def audit_seasons():
    seasons = ["20202021", "20212022", "20222023", "20232024", "20242025"]
    
    results = {}
    
    for season in seasons:
        print(f"\nAuditing {season}...")
        full_schedule = get_full_season_schedule(season)
        completed_games = {gid: g for gid, g in full_schedule.items() if g.get("gameState") == "OFF"}
        
        raw_dir = ROOT / "raw" / season
        local_games = set()
        if raw_dir.exists():
            local_games = {int(d.name) for d in raw_dir.iterdir() if d.is_dir()}
        
        missing = set(completed_games.keys()) - local_games
        
        results[season] = {
            "total_completed": len(completed_games),
            "local_count": len(local_games),
            "missing_count": len(missing),
            "missing_samples": sorted(list(missing))[:10]
        }
        
        print(f"    Completed: {len(completed_games)}")
        print(f"    Local: {len(local_games)}")
        print(f"    Missing: {len(missing)}")

    print("\n" + "="*50)
    print("AUDIT SUMMARY")
    print("="*50)
    for season, data in results.items():
        print(f"{season}: {data['local_count']}/{data['total_completed']} (Missing: {data['missing_count']})")
        if data['missing_count'] > 0:
            print(f"  Samples: {data['missing_samples']}")

if __name__ == "__main__":
    audit_seasons()

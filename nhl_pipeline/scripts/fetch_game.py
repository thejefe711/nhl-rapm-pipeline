#!/usr/bin/env python3
"""
Fetch raw NHL game data from API endpoints.
Saves raw JSON to disk - no transformation, no interpretation.

Two NHL APIs:
- api-web.nhle.com: play-by-play, boxscore
- api.nhle.com/stats/rest: shift charts
"""

import json
import os
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# API endpoints
WEB_API = "https://api-web.nhle.com/v1"
STATS_API = "https://api.nhle.com/stats/rest/en"


@dataclass
class GameFetchResult:
    game_id: int
    season: str
    success: bool
    pbp_path: Optional[str] = None
    shifts_path: Optional[str] = None
    boxscore_path: Optional[str] = None
    error: Optional[str] = None


def fetch_json(url: str, retries: int = 3, delay: float = 2.0) -> Optional[Dict[Any, Any]]:
    """Fetch JSON with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                print(f"  Rate limited, waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                print(f"  404 Not Found: {url}")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  Request failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def fetch_play_by_play(game_id: int) -> Optional[Dict]:
    """Fetch play-by-play from web API."""
    url = f"{WEB_API}/gamecenter/{game_id}/play-by-play"
    print(f"  Fetching play-by-play: {url}")
    return fetch_json(url)


def fetch_boxscore(game_id: int) -> Optional[Dict]:
    """Fetch boxscore from web API."""
    url = f"{WEB_API}/gamecenter/{game_id}/boxscore"
    print(f"  Fetching boxscore: {url}")
    return fetch_json(url)


def fetch_shifts(game_id: int) -> Optional[Dict]:
    """Fetch shifts from stats API (different endpoint!)."""
    url = f"{STATS_API}/shiftcharts?cayenneExp=gameId={game_id}"
    print(f"  Fetching shifts: {url}")
    return fetch_json(url)


def save_raw_json(data: Dict, path: Path) -> None:
    """Save raw JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def fetch_game(game_id: int, output_dir: Path) -> GameFetchResult:
    """
    Fetch all data for a single game.
    
    Game ID format: SSSSTTNNNN
    - SSSS: Season start year (e.g., 2023 for 2023-24)
    - TT: Game type (02 = regular season, 03 = playoffs)
    - NNNN: Game number
    """
    # Parse game ID to get season
    game_id_str = str(game_id)
    season_start = game_id_str[:4]
    season = f"{season_start}{int(season_start) + 1}"
    
    print(f"\nFetching game {game_id} (season {season})...")
    
    # Create output paths
    game_dir = output_dir / season / str(game_id)
    pbp_path = game_dir / "play_by_play.json"
    shifts_path = game_dir / "shifts.json"
    boxscore_path = game_dir / "boxscore.json"
    
    result = GameFetchResult(game_id=game_id, season=season, success=False)
    
    # Fetch play-by-play
    pbp_data = fetch_play_by_play(game_id)
    if pbp_data is None:
        result.error = "Failed to fetch play-by-play"
        return result
    save_raw_json(pbp_data, pbp_path)
    result.pbp_path = str(pbp_path)
    
    # Fetch boxscore
    boxscore_data = fetch_boxscore(game_id)
    if boxscore_data is None:
        result.error = "Failed to fetch boxscore"
        return result
    save_raw_json(boxscore_data, boxscore_path)
    result.boxscore_path = str(boxscore_path)
    
    # Fetch shifts (different API!)
    shifts_data = fetch_shifts(game_id)
    if shifts_data is None:
        result.error = "Failed to fetch shifts"
        return result
    save_raw_json(shifts_data, shifts_path)
    result.shifts_path = str(shifts_path)
    
    result.success = True
    print(f"  OK Successfully fetched game {game_id}")
    return result


def get_sample_game_ids() -> Dict[str, int]:
    """
    Return one sample game ID per season for the last 5 seasons.
    These are mid-season regular season games (game type 02).
    
    Format: YYYYTTNNNN where:
    - YYYY = season start year
    - TT = 02 (regular season)
    - NNNN = game number
    
    Using different game numbers to test API stability across years.
    """
    return {
        "20242025": 2024020415,  # 2024-25 season (recent)
        "20232024": 2023020612,  # 2023-24 season
        "20222023": 2022020508,  # 2022-23 season
        "20212022": 2021020423,  # 2021-22 season
        "20202021": 2020020312,  # 2020-21 season (COVID shortened)
    }


def main():
    """Fetch sample games from each of the last 5 seasons."""
    output_dir = Path(__file__).parent.parent / "raw"
    
    print("=" * 60)
    print("NHL Data Pipeline - Raw Data Fetch")
    print("=" * 60)
    
    sample_games = get_sample_game_ids()
    results = []
    
    for season, game_id in sample_games.items():
        result = fetch_game(game_id, output_dir)
        results.append(result)
        time.sleep(1)  # Be nice to the API
    
    # Summary
    print("\n" + "=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.success)
    print(f"Successful: {success_count}/{len(results)}")
    
    for r in results:
        status = "OK" if r.success else "FAIL"
        error = f" - {r.error}" if r.error else ""
        print(f"  {status} {r.season} (game {r.game_id}){error}")
    
    return results


if __name__ == "__main__":
    main()

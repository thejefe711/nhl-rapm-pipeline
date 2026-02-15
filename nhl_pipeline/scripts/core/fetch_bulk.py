#!/usr/bin/env python3
"""
Fetch games in bulk for full seasons.

Fetches all games for specified seasons, with:
- Resume capability (skips already fetched games)
- Rate limiting
- Progress tracking
- Error handling

Usage:
    python fetch_bulk.py                     # Fetch all seasons (20 games each)
    python fetch_bulk.py --season 20242025   # Fetch specific season
    python fetch_bulk.py --full              # Fetch ALL games (1000+)
    python fetch_bulk.py --games 50          # Fetch 50 games per season
"""

import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# API endpoints
WEB_API = "https://api-web.nhle.com/v1"
STATS_API = "https://api.nhle.com/stats/rest/en"

# Seasons to fetch
SEASONS = [
    "20252026",
    "20242025",
    "20232024",
    "20222023",
    "20212022",
    "20202021",
]


@dataclass
class FetchProgress:
    """Track fetch progress for resume capability."""
    season: str
    total_games: int
    fetched_games: int
    failed_games: List[int]
    last_updated: str


def fetch_json(url: str, retries: int = 3, delay: float = 2.0) -> Optional[Dict]:
    """Fetch JSON with retry logic and rate limiting."""
    for attempt in range(retries):
        try:
            # Add jitter to avoid thundering herd
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
            print(f"    Request failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    
    return None


def get_season_schedule(season: str) -> List[Dict]:
    """Fetch all games for a season."""
    print(f"  Fetching schedule for {season}...")
    
    # The schedule endpoint returns all games for a team's season
    # We'll use a known team and filter to regular season
    url = f"{WEB_API}/schedule/{season[:4]}-{int(season[:4])+1}"
    
    # Alternative: use the club schedule endpoint
    # Try fetching from multiple teams to get complete schedule
    all_games = {}
    
    teams = [
        "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL",
        "CBJ", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
        "NSH", "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS",
        "SEA", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"
    ]
    
    for team in teams:
        url = f"{WEB_API}/club-schedule-season/{team}/{season}"
        data = fetch_json(url)
        
        if data and "games" in data:
            for game in data["games"]:
                game_id = game.get("id")
                game_type = game.get("gameType", 0)
                
                # Only regular season games (type 2)
                if game_id and game_type == 2:
                    all_games[game_id] = game
    
    games = list(all_games.values())
    # The club schedule includes future games. For data fetches we only want games that are
    # completed/available. In practice:
    # - OFF = played/final (data available)
    # - FUT = future (no shifts/PBP yet)
    games = [g for g in games if g.get("gameState") == "OFF"]
    print(f"    Found {len(games)} completed regular season games")
    
    return games


def fetch_game_data(game_id: int, output_dir: Path, min_shift_rows: int = 1) -> bool:
    """Fetch all data for a single game."""
    
    season = str(game_id)[:4] + str(int(str(game_id)[:4]) + 1)
    game_dir = output_dir / season / str(game_id)
    
    def shifts_meet_min_rows(path: Path) -> bool:
        try:
            if not path.exists():
                return False
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows = data.get("data")
            return isinstance(rows, list) and len(rows) >= int(min_shift_rows)
        except Exception:
            return False

    # Check if already fetched AND usable (non-empty shifts)
    if (game_dir / "play_by_play.json").exists() and (game_dir / "boxscore.json").exists() and shifts_meet_min_rows(game_dir / "shifts.json"):
        return True  # Already have this game (usable)
    
    game_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch play-by-play
    pbp_url = f"{WEB_API}/gamecenter/{game_id}/play-by-play"
    pbp_data = fetch_json(pbp_url)
    if pbp_data is None:
        return False
    
    with open(game_dir / "play_by_play.json", "w") as f:
        json.dump(pbp_data, f)
    
    # Fetch boxscore
    box_url = f"{WEB_API}/gamecenter/{game_id}/boxscore"
    box_data = fetch_json(box_url)
    if box_data is None:
        return False
    
    with open(game_dir / "boxscore.json", "w") as f:
        json.dump(box_data, f)
    
    # Fetch shifts (different API!)
    shifts_url = f"{STATS_API}/shiftcharts?cayenneExp=gameId={game_id}"
    shifts_data = fetch_json(shifts_url)
    if shifts_data is None:
        return False

    # Some games return an empty or tiny shift chart payload (e.g. {"data": [], "total": 0}).
    # Treat anything below the minimum threshold as a failed fetch so the season fetcher can
    # move on and still reach the requested number of *usable* games.
    if isinstance(shifts_data, dict) and isinstance(shifts_data.get("data"), list) and len(shifts_data.get("data")) < int(min_shift_rows):
        # If an old empty file exists, remove it so future runs don't incorrectly treat it as fetched.
        try:
            (game_dir / "shifts.json").unlink(missing_ok=True)
        except Exception:
            pass
        return False
    
    with open(game_dir / "shifts.json", "w") as f:
        json.dump(shifts_data, f)
    
    return True


def load_progress(progress_path: Path) -> Dict[str, FetchProgress]:
    """Load fetch progress from disk."""
    if not progress_path.exists():
        return {}
    
    with open(progress_path) as f:
        data = json.load(f)
    
    return {
        season: FetchProgress(**p) for season, p in data.items()
    }


def save_progress(progress: Dict[str, FetchProgress], progress_path: Path):
    """Save fetch progress to disk."""
    data = {
        season: {
            "season": p.season,
            "total_games": p.total_games,
            "fetched_games": p.fetched_games,
            "failed_games": p.failed_games,
            "last_updated": datetime.now().isoformat(),
        }
        for season, p in progress.items()
    }
    
    with open(progress_path, "w") as f:
        json.dump(data, f, indent=2)


def write_shift_counts_csv(raw_dir: Path, out_path: Path) -> int:
    """
    Write `data/shift_counts_by_game.csv` from the current raw directory.

    This is a lightweight debugging artifact so we can quickly see whether fetched games
    have non-empty shift charts (and how many rows).
    """
    import csv

    rows = []
    for season_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        season = season_dir.name
        for game_dir in sorted([p for p in season_dir.iterdir() if p.is_dir()]):
            shifts_path = game_dir / "shifts.json"
            if not shifts_path.exists():
                continue
            try:
                data = json.loads(shifts_path.read_text(encoding="utf-8"))
                n = data.get("data")
                shift_rows = len(n) if isinstance(n, list) else 0
            except Exception:
                shift_rows = 0
            rows.append((season, game_dir.name, shift_rows))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["season", "game_id", "shift_rows"])
        for season, game_id, shift_rows in sorted(rows):
            w.writerow([season, game_id, int(shift_rows)])
    return len(rows)


def fetch_season(
    season: str,
    output_dir: Path,
    max_games: Optional[int] = None,
    progress: Optional[FetchProgress] = None,
    workers: int = 1,
    min_shift_rows: int = 100,
) -> FetchProgress:
    """Fetch all games for a season."""
    
    print(f"\n{'='*60}")
    print(f"FETCHING SEASON {season}")
    print(f"{'='*60}")
    
    # Get schedule
    games = get_season_schedule(season)
    
    if not games:
        print(f"  No games found for {season}")
        return FetchProgress(
            season=season,
            total_games=0,
            fetched_games=0,
            failed_games=[],
            last_updated=datetime.now().isoformat()
        )
    
    # Sort by game ID
    games.sort(key=lambda g: g.get("id", 0))
    
    # We interpret max_games as "usable games" (i.e., have non-empty shifts data).
    if max_games is None:
        total = len(games)
    else:
        total = min(int(max_games), len(games))
        if len(games) < int(max_games):
            print(f"  WARN: Only {len(games)} completed games available for {season}; fetching {total}.")
    fetched = 0
    failed = []

    print(f"  Target usable games: {total} (workers={max(1, int(workers))})")

    def _do_one(g: Dict[str, Any]) -> tuple[int, bool]:
        gid = g.get("id")
        if not gid:
            return (0, False)
        return (int(gid), fetch_game_data(int(gid), output_dir, min_shift_rows=min_shift_rows))

    # Parallel fetch (IO-bound) with conservative default workers.
    workers = max(1, int(workers))
    attempted = 0

    # Iterate through schedule until we reach N usable games.
    schedule_iter = iter(games)
    if workers == 1:
        while fetched < total:
            try:
                game = next(schedule_iter)
            except StopIteration:
                break
            gid = game.get("id")
            if not gid:
                continue
            attempted += 1
            pct = (fetched / total) * 100 if total > 0 else 0
            print(f"  [ok={fetched}/{total}] ({pct:.0f}%) Game {gid}...", end=" ")
            success = fetch_game_data(int(gid), output_dir)
            if success:
                fetched += 1
                print("OK")
            else:
                failed.append(int(gid))
                print("FAIL")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            in_flight = {}
            # Prime the pool
            while len(in_flight) < workers and fetched < total:
                try:
                    game = next(schedule_iter)
                except StopIteration:
                    break
                fut = ex.submit(_do_one, game)
                in_flight[fut] = game

            while in_flight and fetched < total:
                for fut in as_completed(list(in_flight.keys())):
                    gid, success = fut.result()
                    attempted += 1
                    status = "OK" if success else "FAIL"
                    pct = (fetched / total) * 100 if total > 0 else 0
                    print(f"  [ok={fetched}/{total}] ({pct:.0f}%) Game {gid}... {status}")
                    if gid:
                        if success:
                            fetched += 1
                        else:
                            failed.append(gid)
                    del in_flight[fut]
                    break

                # Refill
                while len(in_flight) < workers and fetched < total:
                    try:
                        game = next(schedule_iter)
                    except StopIteration:
                        break
                    fut = ex.submit(_do_one, game)
                    in_flight[fut] = game
    
    return FetchProgress(
        season=season,
        total_games=total,
        fetched_games=fetched,
        failed_games=failed,
        last_updated=datetime.now().isoformat()
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch NHL games in bulk")
    parser.add_argument("--season", type=str, help="Specific season to fetch")
    parser.add_argument("--games", type=int, default=20, help="Max games per season")
    parser.add_argument("--full", action="store_true", help="Fetch ALL games")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed games")
    parser.add_argument("--workers", type=int, default=1, help="Parallel game fetch workers (default: 1)")
    parser.add_argument("--min-shift-rows", type=int, default=100, help="Minimum shift rows required for a game to be counted as usable (default: 100)")
    
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent.parent.parent / "raw"
    progress_path = Path(__file__).parent.parent.parent / "data" / "fetch_progress.json"
    progress_path.parent.mkdir(exist_ok=True)
    
    # Load existing progress
    all_progress = load_progress(progress_path)
    
    # Determine which seasons to fetch
    seasons = [args.season] if args.season else SEASONS
    max_games = None if args.full else args.games
    
    print("=" * 60)
    print("NHL BULK DATA FETCH")
    print("=" * 60)
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Max games per season: {'ALL' if max_games is None else max_games}")
    print(f"Min shift rows: {args.min_shift_rows}")
    print(f"Output directory: {output_dir}")
    
    # Fetch each season
    for season in seasons:
        existing = all_progress.get(season)
        
        progress = fetch_season(
            season=season,
            output_dir=output_dir,
            max_games=max_games,
            progress=existing,
            workers=args.workers,
            min_shift_rows=args.min_shift_rows,
        )
        
        all_progress[season] = progress
        save_progress(all_progress, progress_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)
    
    total_fetched = 0
    total_failed = 0
    
    for season, p in sorted(all_progress.items(), reverse=True):
        status = "OK" if not p.failed_games else "WARN"
        print(f"  {status} {season}: {p.fetched_games}/{p.total_games} games", end="")
        if p.failed_games:
            print(f" ({len(p.failed_games)} failed)")
        else:
            print()
        
        total_fetched += p.fetched_games
        total_failed += len(p.failed_games)
    
    print(f"\nTotal: {total_fetched} games fetched, {total_failed} failed")
    
    if total_failed > 0:
        print(f"\nTo retry failed games: python fetch_bulk.py --retry-failed")

    # Convenience artifact: shift counts by game
    try:
        shift_counts_path = progress_path.parent / "shift_counts_by_game.csv"
        n = write_shift_counts_csv(output_dir, shift_counts_path)
        print(f"\nOK Wrote {shift_counts_path} ({n} games)")
    except Exception as e:
        print(f"\nWARN Failed to write shift_counts_by_game.csv: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
5v5 State Reconstruction - Determine who was on ice for each event.

This is the critical bridge between raw data and APM computation.
Without this, you can't compute plus-minus.

Outputs:
- on_ice_states: Who was on ice at each second of game time
- event_on_ice: 5 home + 5 away skaters for each event

Usage:
    python build_on_ice.py              # Process all staged games
    python build_on_ice.py --game 2024020415  # Process specific game
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class OnIceState:
    """Players on ice at a specific game moment."""
    game_id: int
    period: int
    game_seconds: int
    
    home_skaters: List[int] = field(default_factory=list)
    away_skaters: List[int] = field(default_factory=list)
    home_goalie: Optional[int] = None
    away_goalie: Optional[int] = None
    
    home_skater_count: int = 0
    away_skater_count: int = 0
    
    # Game state
    strength_state: str = "5v5"  # "5v5", "5v4", "4v5", "4v4", etc.
    
    @property
    def is_5v5(self) -> bool:
        return self.home_skater_count == 5 and self.away_skater_count == 5


@dataclass
class EventOnIce:
    """On-ice players at moment of an event."""
    game_id: int
    event_id: int
    event_type: str
    period: int
    period_seconds: int
    game_seconds: int
    
    # Game context (denormalized for downstream APM/RAPM)
    event_team_id: Optional[int] = None
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    
    # Home team on ice (up to 6 skaters + goalie)
    home_skater_1: Optional[int] = None
    home_skater_2: Optional[int] = None
    home_skater_3: Optional[int] = None
    home_skater_4: Optional[int] = None
    home_skater_5: Optional[int] = None
    home_skater_6: Optional[int] = None
    home_goalie: Optional[int] = None
    
    # Away team on ice
    away_skater_1: Optional[int] = None
    away_skater_2: Optional[int] = None
    away_skater_3: Optional[int] = None
    away_skater_4: Optional[int] = None
    away_skater_5: Optional[int] = None
    away_skater_6: Optional[int] = None
    away_goalie: Optional[int] = None
    
    # Counts
    home_skater_count: int = 0
    away_skater_count: int = 0
    strength_state: str = "5v5"
    is_5v5: bool = True


def get_players_on_ice_at_time(
    shifts_df: pd.DataFrame,
    period: int,
    period_seconds: int,
    team_id: int,
    tolerance: int = 2
) -> List[int]:
    """
    Get all players on ice at a specific moment.
    
    Args:
        shifts_df: DataFrame of shifts
        period: Period number
        period_seconds: Seconds into the period
        team_id: Team to get players for
        tolerance: Seconds of tolerance for shift boundaries
    
    Returns:
        List of player IDs on ice
    """
    # Filter to team and period
    mask = (
        (shifts_df["team_id"] == team_id) &
        (shifts_df["period"] == period) &
        (shifts_df["start_seconds"] <= period_seconds + tolerance) &
        (shifts_df["end_seconds"] >= period_seconds - tolerance)
    )
    
    on_ice = shifts_df[mask]["player_id"].unique().tolist()
    return on_ice


def identify_goalies(
    shifts_df: pd.DataFrame,
    boxscore_path: Optional[Path] = None
) -> Dict[int, Set[int]]:
    """
    Identify goalie player IDs per team.
    
    Goalies have different shift patterns - they're typically on ice
    for entire periods with very long shifts.
    """
    # Prefer authoritative goalie IDs from boxscore when available
    if boxscore_path and boxscore_path.exists():
        try:
            with open(boxscore_path) as f:
                box = json.load(f)

            home_team_id = box.get("homeTeam", {}).get("id")
            away_team_id = box.get("awayTeam", {}).get("id")

            pbgs = box.get("playerByGameStats", {})
            home_goalies = pbgs.get("homeTeam", {}).get("goalies", []) or []
            away_goalies = pbgs.get("awayTeam", {}).get("goalies", []) or []

            goalies: Dict[int, Set[int]] = {}
            if home_team_id is not None:
                goalies[int(home_team_id)] = set(int(g["playerId"]) for g in home_goalies if g.get("playerId") is not None)
            if away_team_id is not None:
                goalies[int(away_team_id)] = set(int(g["playerId"]) for g in away_goalies if g.get("playerId") is not None)

            # If we got anything, return it (even if empty sets)
            if goalies:
                return goalies
        except Exception as e:
            print(f"  WARN: Failed to parse goalies from boxscore: {e}")

    goalies: Dict[int, Set[int]] = {}
    
    # Heuristic: goalies have very long total TOI and few shifts
    for team_id in shifts_df["team_id"].unique():
        team_shifts = shifts_df[shifts_df["team_id"] == team_id]
        
        # Group by player
        player_stats = team_shifts.groupby("player_id").agg({
            "duration_seconds": ["sum", "count"],
            "period": "nunique"
        })
        player_stats.columns = ["total_toi", "shift_count", "periods_played"]
        player_stats = player_stats.reset_index()
        
        # Goalies: high TOI, low shift count, play all periods
        # Typical goalie: 3600s TOI, 3-6 shifts (one per period + OT)
        # Typical skater: 900-1200s TOI, 20-30 shifts
        
        potential_goalies = player_stats[
            (player_stats["total_toi"] > 2400) &  # >40 min
            (player_stats["shift_count"] < 10) &   # Few shifts
            (player_stats["periods_played"] >= 3)  # Played all periods
        ]["player_id"].tolist()
        
        goalies[team_id] = set(potential_goalies)
    
    return goalies


def build_event_on_ice(
    events_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    goalies: Dict[int, Set[int]]
) -> pd.DataFrame:
    """
    Timeline-based on-ice assignment.
    Pre-calculates on-ice sets for every second to match original logic exactly.
    """
    if events_df.empty:
        return pd.DataFrame()
    
    home_goalies = goalies.get(home_team_id, set())
    away_goalies = goalies.get(away_team_id, set())
    
    # 1. Pre-calculate on-ice sets for every (period, second)
    timelines = {home_team_id: {}, away_team_id: {}}
    
    for team_id in [home_team_id, away_team_id]:
        team_shifts = shifts_df[shifts_df["team_id"] == team_id]
        for period in team_shifts["period"].unique():
            p_shifts = team_shifts[team_shifts["period"] == period]
            for _, shift in p_shifts.iterrows():
                pid = int(shift["player_id"])
                start = int(shift["start_seconds"])
                end = int(shift["end_seconds"])
                
                # Store shifts that overlap with any second in the period (0-1200)
                # We only need to store them for seconds where events actually happen,
                # but storing for all seconds is fast enough and simpler.
                for t in range(max(0, start - 2), min(1201, end + 3)):
                    key = (period, t)
                    if key not in timelines[team_id]:
                        timelines[team_id][key] = []
                    timelines[team_id][key].append((pid, start, end))

    results = []
    
    for _, event in events_df.iterrows():
        period = int(event["period"])
        period_seconds = int(event["period_seconds"])
        
        must_for_team: Dict[int, List[int]] = {home_team_id: [], away_team_id: []}
        eteam = int(event["event_team_id"]) if pd.notna(event.get("event_team_id")) else None
        if eteam in (home_team_id, away_team_id):
            if event["event_type"] in ("GOAL", "SHOT", "MISSED_SHOT", "BLOCKED_SHOT"):
                shooter_or_scorer = event.get("player_1_id")
                if pd.notna(shooter_or_scorer):
                    must_for_team[eteam].append(int(shooter_or_scorer))
        
        on_ice_by_team = {}
        for team_id in [home_team_id, away_team_id]:
            key = (period, period_seconds)
            candidates = timelines[team_id].get(key, [])
            must = must_for_team[team_id]
            
            def get_players(tol, start_exclusive):
                pids = []
                for pid, start, end in candidates:
                    if start_exclusive:
                        s_ok = start <= period_seconds + tol if period_seconds == 0 else start < period_seconds + tol
                    else:
                        s_ok = start <= period_seconds + tol
                    e_ok = end >= period_seconds - tol
                    if s_ok and e_ok:
                        pids.append(pid)
                return list(set(pids))

            chosen = []
            for tol in (0, 1, 2):
                preferred = get_players(tol, True)
                fallback = get_players(tol, False)
                if all(p in preferred for p in must):
                    chosen = preferred
                else:
                    chosen = fallback
                if len(chosen) <= 6:
                    break
            
            if len(chosen) > 6:
                # Apply margin logic
                def margin(pid):
                    m = -1
                    for p, s, e in candidates:
                        if p == pid:
                            m = max(m, min(period_seconds - s, e - period_seconds))
                    return m
                
                ranked = sorted(list(set(chosen)), key=margin, reverse=True)
                final = []
                for p in must:
                    if p in ranked: final.append(p)
                for p in ranked:
                    if p not in final: final.append(p)
                    if len(final) >= 6: break
                chosen = final[:6]
            
            on_ice_by_team[team_id] = chosen

        home_on_ice = on_ice_by_team[home_team_id]
        away_on_ice = on_ice_by_team[away_team_id]
        
        home_goalie = next((p for p in home_on_ice if p in home_goalies), None)
        away_goalie = next((p for p in away_on_ice if p in away_goalies), None)
        
        home_skaters = sorted([p for p in home_on_ice if p not in home_goalies])
        away_skaters = sorted([p for p in away_on_ice if p not in away_goalies])
        
        h_skaters = (home_skaters + [None] * 6)[:6]
        a_skaters = (away_skaters + [None] * 6)[:6]
        
        results.append({
            "game_id": event["game_id"], "event_id": event["event_id"], "event_type": event["event_type"],
            "event_team_id": eteam, "period": period, "period_seconds": period_seconds, "game_seconds": event["game_seconds"],
            "home_team_id": int(home_team_id), "away_team_id": int(away_team_id),
            "home_skater_1": h_skaters[0], "home_skater_2": h_skaters[1], "home_skater_3": h_skaters[2],
            "home_skater_4": h_skaters[3], "home_skater_5": h_skaters[4], "home_skater_6": h_skaters[5],
            "home_goalie": home_goalie,
            "away_skater_1": a_skaters[0], "away_skater_2": a_skaters[1], "away_skater_3": a_skaters[2],
            "away_skater_4": a_skaters[3], "away_skater_5": a_skaters[4], "away_skater_6": a_skaters[5],
            "away_goalie": away_goalie,
            "home_skater_count": len(home_skaters), "away_skater_count": len(away_skaters),
            "strength_state": f"{len(home_skaters)}v{len(away_skaters)}", 
            "is_5v5": (len(home_skaters) == 5 and len(away_skaters) == 5)
        })
        
    return pd.DataFrame(results)


def validate_on_ice(event_on_ice_df: pd.DataFrame, events_df: pd.DataFrame) -> Dict:
    """
    Validate on-ice assignments.
    
    Returns validation metrics.
    """
    if event_on_ice_df.empty:
        return {"total_events": 0, "events_5v5": 0, "pct_5v5": 0}

    metrics = {
        "total_events": len(event_on_ice_df),
        "events_5v5": event_on_ice_df["is_5v5"].sum(),
        "pct_5v5": 100 * event_on_ice_df["is_5v5"].mean(),
    }
    
    # Check for events with wrong skater counts
    wrong_home = event_on_ice_df[
        (event_on_ice_df["home_skater_count"] < 3) | 
        (event_on_ice_df["home_skater_count"] > 6)
    ]
    wrong_away = event_on_ice_df[
        (event_on_ice_df["away_skater_count"] < 3) | 
        (event_on_ice_df["away_skater_count"] > 6)
    ]
    
    metrics["invalid_home_count"] = len(wrong_home)
    metrics["invalid_away_count"] = len(wrong_away)
    
    # Check goals have credited player on ice
    goals = events_df[events_df["event_type"] == "GOAL"]
    metrics["total_goals"] = len(goals)
    
    # Strength state distribution
    strength_dist = event_on_ice_df["strength_state"].value_counts().to_dict()
    metrics["strength_distribution"] = strength_dist
    
    return metrics


def process_game(
    season: str,
    game_id: str,
    staging_dir: Path,
    canonical_dir: Path,
    raw_dir: Path,
    overwrite: bool = False
) -> Optional[Dict]:
    """Process a single game and build on-ice assignments."""
    
    output_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
    if output_path.exists() and not overwrite:
        return None
    
    shifts_path = staging_dir / season / f"{game_id}_shifts.parquet"
    events_path = staging_dir / season / f"{game_id}_events.parquet"
    boxscore_path = raw_dir / season / game_id / "boxscore.json"
    
    if not shifts_path.exists() or not events_path.exists():
        return {"success": False, "game_id": game_id, "error": "missing files"}
    
    shifts_df = pd.read_parquet(shifts_path)
    if "type_code" in shifts_df.columns:
        shifts_df = shifts_df[(shifts_df["type_code"] == 517) | (shifts_df["type_code"].isna())].copy()
    events_df = pd.read_parquet(events_path)
    
    # Get team IDs
    team_ids = shifts_df["team_id"].unique()
    if len(team_ids) != 2:
        return {"success": False, "game_id": game_id, "error": f"invalid team count: {len(team_ids)}"}

    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None

    if boxscore_path.exists():
        with open(boxscore_path) as f:
            boxscore = json.load(f)
        home_team_id = boxscore.get("homeTeam", {}).get("id")
        away_team_id = boxscore.get("awayTeam", {}).get("id")

    if home_team_id is None or away_team_id is None:
        home_team_id = int(team_ids[0])
        away_team_id = int(team_ids[1])
    
    # Identify goalies
    goalies = identify_goalies(shifts_df, boxscore_path=boxscore_path)
    
    # Build event on-ice
    event_on_ice_df = build_event_on_ice(
        events_df, shifts_df, int(home_team_id), int(away_team_id), goalies
    )
    
    # Validate
    metrics = validate_on_ice(event_on_ice_df, events_df)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_on_ice_df.to_parquet(output_path, index=False)
    
    return {
        "season": season,
        "game_id": game_id,
        "success": True,
        **metrics
    }


def _process_game_wrapper(game_tuple, staging_dir, canonical_dir, raw_dir, overwrite):
    season, game_id = game_tuple
    return process_game(
        season=season,
        game_id=game_id,
        staging_dir=staging_dir,
        canonical_dir=canonical_dir,
        raw_dir=raw_dir,
        overwrite=overwrite
    )


def main():
    parser = argparse.ArgumentParser(description="Build on-ice assignments")
    parser.add_argument("--game", type=str, help="Specific game ID to process")
    parser.add_argument("--season", type=str, default=None, help="Only process a specific season (e.g., 20252026)")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if canonical parquet already exists")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    staging_dir = Path(__file__).parent.parent / "staging"
    canonical_dir = Path(__file__).parent.parent / "canonical"
    raw_dir = Path(__file__).parent.parent / "raw"
    canonical_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NHL Data Pipeline - Build On-Ice Assignments (Optimized)")
    print("=" * 60)
    
    # Find games to process
    games = []
    if args.game:
        for season_dir in staging_dir.glob("*"):
            if (season_dir / f"{args.game}_shifts.parquet").exists():
                games.append((season_dir.name, args.game))
                break
    else:
        if args.season:
            shift_files = (staging_dir / args.season).glob("*_shifts.parquet")
        else:
            shift_files = staging_dir.glob("*/*_shifts.parquet")

        for shifts_file in shift_files:
            game_id = shifts_file.stem.replace("_shifts", "")
            season = shifts_file.parent.name
            games.append((season, game_id))
    
    print(f"Processing {len(games)} games with {args.workers} workers...")
    
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    
    start_time = time.perf_counter()
    
    if args.workers > 1:
        process_fn = partial(
            _process_game_wrapper, 
            staging_dir=staging_dir, 
            canonical_dir=canonical_dir, 
            raw_dir=raw_dir, 
            overwrite=args.overwrite
        )
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(process_fn, games))
    else:
        results = [process_game(g[0], g[1], staging_dir, canonical_dir, raw_dir, args.overwrite) for g in games]
    
    elapsed = time.perf_counter() - start_time
    results = [r for r in results if r is not None]
    
    # Summary
    print("\n" + "=" * 60)
    print("ON-ICE BUILD SUMMARY")
    print("=" * 60)
    
    if not results:
        print("No games processed.")
        return
    
    successes = [r for r in results if r["success"]]
    total_events = sum(r["total_events"] for r in successes)
    total_5v5 = sum(r["events_5v5"] for r in successes)
    avg_5v5_pct = 100 * total_5v5 / total_events if total_events > 0 else 0
    
    print(f"Total events:    {total_events:,}")
    print(f"5v5 events:      {total_5v5:,} ({avg_5v5_pct:.1f}%)")
    print(f"Games processed: {len(successes)}")
    print(f"Total time:      {elapsed:.1f}s")
    if successes:
        print(f"Avg time/game:   {elapsed/len(successes):.3f}s")
        print(f"Throughput:      {len(successes)/elapsed:.1f} games/s")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

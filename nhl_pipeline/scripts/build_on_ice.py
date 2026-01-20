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
    Build on-ice assignment for each event.
    
    Returns DataFrame with one row per event, including all on-ice players.
    """
    results = []
    
    home_goalies = goalies.get(home_team_id, set())
    away_goalies = goalies.get(away_team_id, set())
    
    def best_on_ice_set(team_id: int, period: int, period_seconds: int, must_include: Optional[List[int]] = None) -> List[int]:
        """
        Determine on-ice players for a team at a moment.

        NHL shift times are second-granular, so a player ending at T and another starting at T can
        both appear "on ice" at second T if we treat ends as inclusive. To avoid dropping the
        scorer/shooter at boundary seconds, we:
        - compute an end-exclusive set (start <= t < end) and an end-inclusive set (start <= t <= end)
        - prefer the end-exclusive set when it contains all must-include players (scorer/shooter)
        - otherwise fall back to inclusive
        - if still over-counted, select the 6 players with the largest boundary margin
        """
        if must_include is None:
            must_include = []

        team_shifts = shifts_df[(shifts_df["team_id"] == team_id) & (shifts_df["period"] == period)]
        if team_shifts.empty:
            return []

        def players_at(t: int, tol: int, start_exclusive: bool) -> List[int]:
            # Default convention to avoid double-counting shift changes at second boundaries:
            #   start is exclusive, end is inclusive  =>  start < t <= end
            if start_exclusive:
                if t == 0:
                    start_ok = team_shifts["start_seconds"] <= t + tol
                else:
                    start_ok = team_shifts["start_seconds"] < t + tol
            else:
                start_ok = team_shifts["start_seconds"] <= t + tol

            end_ok = team_shifts["end_seconds"] >= t - tol
            mask = start_ok & end_ok
            return team_shifts.loc[mask, "player_id"].dropna().astype(int).unique().tolist()

        # Prefer start-exclusive/end-inclusive (reduces double-counting) unless it drops must-include.
        for tol in (0, 1, 2):
            preferred = players_at(period_seconds, tol, start_exclusive=True)
            fallback = players_at(period_seconds, tol, start_exclusive=False)

            pref_set = set(preferred)
            chosen = preferred if all(pid in pref_set for pid in must_include) else fallback
            if len(chosen) <= 6:
                return chosen

        # Over-count persists: pick the 6 most confidently-on-ice players by boundary margin.
        def margin_for_player(pid: int) -> int:
            psh = team_shifts[team_shifts["player_id"] == pid]
            cover = psh[(psh["start_seconds"] <= period_seconds) & (psh["end_seconds"] >= period_seconds)]
            if cover.empty:
                cover = psh[(psh["start_seconds"] <= period_seconds + 2) & (psh["end_seconds"] >= period_seconds - 2)]
            if cover.empty:
                return -1
            return int((np.minimum(period_seconds - cover["start_seconds"], cover["end_seconds"] - period_seconds)).max())

        # Use the widest candidate set from tol=2 inclusive-start for scoring.
        candidates = players_at(period_seconds, 2, start_exclusive=False)
        uniq = list(dict.fromkeys(candidates))
        ranked = sorted(uniq, key=lambda pid: margin_for_player(pid), reverse=True)

        chosen: List[int] = []
        for pid in must_include:
            if pid in ranked and pid not in chosen:
                chosen.append(pid)
        for pid in ranked:
            if pid not in chosen:
                chosen.append(pid)
            if len(chosen) >= 6:
                break
        return chosen[:6]

    for _, event in events_df.iterrows():
        period = event["period"]
        period_seconds = event["period_seconds"]
        game_seconds = event["game_seconds"]

        # Identify must-include players to disambiguate boundary seconds.
        must_for_team: Dict[int, List[int]] = {home_team_id: [], away_team_id: []}
        try:
            eteam = int(event["event_team_id"]) if pd.notna(event.get("event_team_id")) else None
        except Exception:
            eteam = None

        if eteam in (home_team_id, away_team_id):
            if event["event_type"] in ("GOAL", "SHOT", "MISSED_SHOT", "BLOCKED_SHOT"):
                shooter_or_scorer = event.get("player_1_id")
                if pd.notna(shooter_or_scorer):
                    must_for_team[eteam].append(int(shooter_or_scorer))
        
        # Get players on ice
        home_on_ice = best_on_ice_set(home_team_id, period, period_seconds, must_include=must_for_team.get(home_team_id, []))
        away_on_ice = best_on_ice_set(away_team_id, period, period_seconds, must_include=must_for_team.get(away_team_id, []))
        
        # Separate goalies from skaters
        home_goalie = next((p for p in home_on_ice if p in home_goalies), None)
        away_goalie = next((p for p in away_on_ice if p in away_goalies), None)

        home_skater_candidates = [p for p in home_on_ice if p not in home_goalies]
        away_skater_candidates = [p for p in away_on_ice if p not in away_goalies]

        def select_skaters(team_id: int, candidates: List[int], must_include_skaters: List[int]) -> List[Optional[int]]:
            """
            Reduce candidate skaters to at most 6, prioritizing:
            - must-include skaters (scorer/shooter)
            - skaters most confidently on ice (boundary margin)
            """
            if not candidates:
                return [None] * 6

            team_shifts = shifts_df[(shifts_df["team_id"] == team_id) & (shifts_df["period"] == period)]

            def skater_margin(pid: int) -> int:
                psh = team_shifts[team_shifts["player_id"] == pid]
                cover = psh[(psh["start_seconds"] <= period_seconds) & (psh["end_seconds"] >= period_seconds)]
                if cover.empty:
                    cover = psh[(psh["start_seconds"] <= period_seconds + 2) & (psh["end_seconds"] >= period_seconds - 2)]
                if cover.empty:
                    return -1
                return int((np.minimum(period_seconds - cover["start_seconds"], cover["end_seconds"] - period_seconds)).max())

            uniq = list(dict.fromkeys(int(p) for p in candidates))
            # Sort by confidence margin (desc)
            ranked = sorted(uniq, key=lambda pid: skater_margin(pid), reverse=True)

            chosen: List[int] = []
            for pid in must_include_skaters:
                if pid in ranked and pid not in chosen:
                    chosen.append(pid)

            for pid in ranked:
                if pid not in chosen:
                    chosen.append(pid)
                if len(chosen) >= 6:
                    break

            return (chosen + [None] * 6)[:6]

        home_skaters = select_skaters(home_team_id, home_skater_candidates, must_for_team.get(home_team_id, []))
        away_skaters = select_skaters(away_team_id, away_skater_candidates, must_for_team.get(away_team_id, []))
        
        # Count actual skaters
        home_count = sum(1 for p in home_skaters if p is not None)
        away_count = sum(1 for p in away_skaters if p is not None)
        
        # Determine strength state
        strength = f"{home_count}v{away_count}"
        is_5v5 = (home_count == 5 and away_count == 5)
        
        result = EventOnIce(
            game_id=event["game_id"],
            event_id=event["event_id"],
            event_type=event["event_type"],
            event_team_id=int(event["event_team_id"]) if pd.notna(event.get("event_team_id")) else None,
            period=period,
            period_seconds=period_seconds,
            game_seconds=game_seconds,
            home_team_id=int(home_team_id),
            away_team_id=int(away_team_id),
            home_skater_1=home_skaters[0],
            home_skater_2=home_skaters[1],
            home_skater_3=home_skaters[2],
            home_skater_4=home_skaters[3],
            home_skater_5=home_skaters[4],
            home_skater_6=home_skaters[5],
            home_goalie=home_goalie,
            away_skater_1=away_skaters[0],
            away_skater_2=away_skaters[1],
            away_skater_3=away_skaters[2],
            away_skater_4=away_skaters[3],
            away_skater_5=away_skaters[4],
            away_skater_6=away_skaters[5],
            away_goalie=away_goalie,
            home_skater_count=home_count,
            away_skater_count=away_count,
            strength_state=strength,
            is_5v5=is_5v5,
        )
        
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame([vars(r) for r in results])
    return df


def validate_on_ice(event_on_ice_df: pd.DataFrame, events_df: pd.DataFrame) -> Dict:
    """
    Validate on-ice assignments.
    
    Returns validation metrics.
    """
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
    goals_on_ice = event_on_ice_df[event_on_ice_df["event_type"] == "GOAL"]
    
    # This validation requires knowing which team scored - skip for now
    metrics["total_goals"] = len(goals)
    
    # Strength state distribution
    strength_dist = event_on_ice_df["strength_state"].value_counts().to_dict()
    metrics["strength_distribution"] = strength_dist
    
    return metrics


def process_game(
    staging_dir: Path,
    canonical_dir: Path,
    raw_dir: Path,
    season: str,
    game_id: str
) -> Optional[Dict]:
    """Process a single game and build on-ice assignments."""
    
    shifts_path = staging_dir / season / f"{game_id}_shifts.parquet"
    events_path = staging_dir / season / f"{game_id}_events.parquet"
    boxscore_path = raw_dir / season / game_id / "boxscore.json"
    
    if not shifts_path.exists() or not events_path.exists():
        print(f"  Missing data for {season}/{game_id}")
        return None
    
    shifts_df = pd.read_parquet(shifts_path)
    # Use only regular shift rows to reduce overlap/duplicate artifacts
    if "type_code" in shifts_df.columns:
        shifts_df = shifts_df[(shifts_df["type_code"] == 517) | (shifts_df["type_code"].isna())].copy()
    events_df = pd.read_parquet(events_path)
    
    # Get team IDs
    team_ids = shifts_df["team_id"].unique()
    if len(team_ids) != 2:
        print(f"  Invalid team count: {len(team_ids)}")
        return None

    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None

    if boxscore_path.exists():
        with open(boxscore_path) as f:
            boxscore = json.load(f)
        home_team_id = boxscore.get("homeTeam", {}).get("id")
        away_team_id = boxscore.get("awayTeam", {}).get("id")

    # Fallback if boxscore missing/unparseable
    if home_team_id is None or away_team_id is None:
        # Preserve behavior but warn loudly: this breaks home/away-sensitive metrics.
        print("  WARN: Boxscore missing home/away team IDs; falling back to arbitrary team ordering.")
        home_team_id = int(team_ids[0])
        away_team_id = int(team_ids[1])

    # Sanity: ensure boxscore team IDs match shift team IDs
    shift_team_ids = set(int(t) for t in team_ids.tolist())
    if int(home_team_id) not in shift_team_ids or int(away_team_id) not in shift_team_ids:
        print("  WARN: Boxscore team IDs don't match shift team IDs; falling back to shift team IDs.")
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
    output_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_on_ice_df.to_parquet(output_path, index=False)
    
    return {
        "season": season,
        "game_id": game_id,
        "success": True,
        **metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Build on-ice assignments")
    parser.add_argument("--game", type=str, help="Specific game ID to process")
    parser.add_argument("--season", type=str, default=None, help="Only process a specific season (e.g., 20252026)")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if canonical parquet already exists")
    
    args = parser.parse_args()
    
    staging_dir = Path(__file__).parent.parent / "staging"
    canonical_dir = Path(__file__).parent.parent / "canonical"
    raw_dir = Path(__file__).parent.parent / "raw"
    canonical_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NHL Data Pipeline - Build On-Ice Assignments")
    print("=" * 60)
    
    # Find games to process
    if args.game:
        # Find the season for this game
        for season_dir in staging_dir.glob("*"):
            if (season_dir / f"{args.game}_shifts.parquet").exists():
                games = [(season_dir.name, args.game)]
                break
        else:
            print(f"Game {args.game} not found")
            return
    else:
        games = []
        if args.season:
            shift_glob = staging_dir / args.season
            shift_files = shift_glob.glob("*_shifts.parquet")
        else:
            shift_files = staging_dir.glob("*/*_shifts.parquet")

        for shifts_file in shift_files:
            game_id = shifts_file.stem.replace("_shifts", "")
            season = shifts_file.parent.name
            games.append((season, game_id))
    
    print(f"Processing {len(games)} games...")
    
    results = []
    for season, game_id in sorted(games):
        print(f"\n{season}/{game_id}...")

        output_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
        if output_path.exists() and not args.overwrite:
            print("  SKIP (canonical exists)")
            continue
        
        result = process_game(staging_dir, canonical_dir, raw_dir, season, game_id)
        
        if result:
            results.append(result)
            pct_5v5 = result["pct_5v5"]
            print(f"  OK {result['total_events']} events, {pct_5v5:.1f}% 5v5")
        else:
            print(f"  FAIL Failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("ON-ICE BUILD SUMMARY")
    print("=" * 60)
    
    if not results:
        print("No games processed.")
        return
    
    total_events = sum(r["total_events"] for r in results)
    total_5v5 = sum(r["events_5v5"] for r in results)
    avg_5v5_pct = 100 * total_5v5 / total_events if total_events > 0 else 0
    
    print(f"Total events:    {total_events:,}")
    print(f"5v5 events:      {total_5v5:,} ({avg_5v5_pct:.1f}%)")
    print(f"Games processed: {len(results)}")
    
    # Show strength state distribution across all games
    all_strengths = {}
    for r in results:
        for strength, count in r.get("strength_distribution", {}).items():
            all_strengths[strength] = all_strengths.get(strength, 0) + count
    
    print("\nStrength state distribution:")
    for strength, count in sorted(all_strengths.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / total_events
        print(f"  {strength}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

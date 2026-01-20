#!/usr/bin/env python3
"""
Validate On-Ice Assignments for APM Correctness

Before computing APM, we need to verify:
1. Goal scorers are actually on ice when they score
2. Event team attribution is correct
3. Skater counts are valid (5v5 means exactly 5 skaters each)
4. Our computed +/- matches official NHL +/- (within tolerance)

This is the "trust gate" before any plus-minus computation.

Usage:
    python validate_on_ice.py              # Validate all games
    python validate_on_ice.py --game 2024020415  # Specific game
    python validate_on_ice.py --strict     # Fail on any mismatch
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np


@dataclass
class OnIceValidation:
    """Validation results for on-ice assignments."""
    game_id: str
    season: str
    
    # Goal validation
    total_goals: int = 0
    goals_with_scorer_on_ice: int = 0
    goals_missing_scorer: int = 0
    
    # Team attribution
    goals_with_valid_team: int = 0
    goals_with_invalid_team: int = 0
    
    # Skater count validation (for 5v5 events)
    events_5v5: int = 0
    events_5v5_valid_counts: int = 0  # Exactly 5 skaters each side
    events_5v5_invalid_counts: int = 0
    
    # Plus-minus validation (if boxscore available)
    players_checked: int = 0
    players_pm_match: int = 0
    players_pm_mismatch: int = 0
    max_pm_difference: int = 0
    
    # Overall
    all_passed: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_goal_scorer_on_ice(
    events_df: pd.DataFrame,
    event_on_ice_df: pd.DataFrame
) -> Tuple[int, int, List[Dict]]:
    """
    Validate that goal scorers appear in on-ice data.
    
    Returns: (total_goals, goals_with_scorer, missing_details)
    """
    goals = events_df[events_df["event_type"] == "GOAL"].copy()

    # Exclude shootout goals: they have no on-ice shifts and do not count for plus/minus.
    if "period_type" in goals.columns:
        goals = goals[goals["period_type"].astype(str).str.upper() != "SO"].copy()
    
    if goals.empty:
        return 0, 0, []
    
    missing = []
    valid = 0
    
    for _, goal in goals.iterrows():
        event_id = goal["event_id"]
        scorer_id = goal.get("player_1_id")  # Scorer is player_1 for goals
        
        if pd.isna(scorer_id):
            missing.append({
                "event_id": event_id,
                "reason": "No scorer_id in event data"
            })
            continue
        
        scorer_id = int(scorer_id)
        
        # Find this event in on-ice data
        on_ice_row = event_on_ice_df[event_on_ice_df["event_id"] == event_id]
        
        if on_ice_row.empty:
            missing.append({
                "event_id": event_id,
                "scorer_id": scorer_id,
                "reason": "Event not in on-ice data"
            })
            continue
        
        on_ice_row = on_ice_row.iloc[0]
        
        # Check if scorer is in any skater position
        all_skaters = []
        for side in ["home", "away"]:
            for i in range(1, 7):
                skater = on_ice_row.get(f"{side}_skater_{i}")
                if pd.notna(skater):
                    all_skaters.append(int(skater))
        
        if scorer_id in all_skaters:
            valid += 1
        else:
            missing.append({
                "event_id": event_id,
                "scorer_id": scorer_id,
                "period": on_ice_row.get("period"),
                "period_seconds": on_ice_row.get("period_seconds"),
                "skaters_on_ice": all_skaters,
                "reason": "Scorer not in on-ice skaters"
            })
    
    return len(goals), valid, missing


def validate_event_team_attribution(
    events_df: pd.DataFrame,
    shifts_df: pd.DataFrame
) -> Tuple[int, int, List[Dict]]:
    """
    Validate that event_team_id matches the scorer's actual team.
    
    Returns: (total_goals, valid_goals, invalid_details)
    """
    goals = events_df[events_df["event_type"] == "GOAL"].copy()

    # Exclude shootout goals (no on-ice context; not part of +/-).
    if "period_type" in goals.columns:
        goals = goals[goals["period_type"].astype(str).str.upper() != "SO"].copy()
    
    if goals.empty:
        return 0, 0, []
    
    # Build player -> team mapping from shifts
    player_teams = shifts_df.groupby("player_id")["team_id"].first().to_dict()
    
    invalid = []
    valid = 0
    
    for _, goal in goals.iterrows():
        event_id = goal["event_id"]
        event_team_id = goal.get("event_team_id")
        scorer_id = goal.get("player_1_id")
        
        if pd.isna(scorer_id) or pd.isna(event_team_id):
            continue
        
        scorer_id = int(scorer_id)
        event_team_id = int(event_team_id)
        
        scorer_team = player_teams.get(scorer_id)
        
        if scorer_team is None:
            invalid.append({
                "event_id": event_id,
                "scorer_id": scorer_id,
                "event_team_id": event_team_id,
                "reason": "Scorer not found in shifts data"
            })
            continue
        
        if scorer_team == event_team_id:
            valid += 1
        else:
            invalid.append({
                "event_id": event_id,
                "scorer_id": scorer_id,
                "scorer_team": scorer_team,
                "event_team_id": event_team_id,
                "reason": "event_team_id does not match scorer's team"
            })
    
    return len(goals), valid, invalid


def validate_5v5_skater_counts(
    event_on_ice_df: pd.DataFrame
) -> Tuple[int, int, List[Dict]]:
    """
    Validate that 5v5 events have exactly 5 skaters per side.
    
    Returns: (total_5v5, valid_5v5, invalid_details)
    """
    events_5v5 = event_on_ice_df[event_on_ice_df["is_5v5"] == True].copy()
    
    if events_5v5.empty:
        return 0, 0, []
    
    invalid = []
    valid = 0
    
    for _, event in events_5v5.iterrows():
        home_count = event.get("home_skater_count", 0)
        away_count = event.get("away_skater_count", 0)
        
        if home_count == 5 and away_count == 5:
            valid += 1
        else:
            invalid.append({
                "event_id": event["event_id"],
                "home_skater_count": home_count,
                "away_skater_count": away_count,
                "strength_state": event.get("strength_state"),
                "reason": f"5v5 flagged but counts are {home_count}v{away_count}"
            })
    
    return len(events_5v5), valid, invalid


def compute_simple_plus_minus(
    event_on_ice_df: pd.DataFrame,
    events_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    only_5v5: bool = False
) -> Dict[int, int]:
    """
    Compute simple plus-minus from on-ice data.
    
    Returns: {player_id: plus_minus}
    """
    # Get goals
    goals_on_ice = event_on_ice_df[event_on_ice_df["event_type"] == "GOAL"].copy()
    
    if only_5v5:
        goals_on_ice = goals_on_ice[goals_on_ice["is_5v5"] == True]
    
    if goals_on_ice.empty:
        return {}
    
    # Ensure we have an authoritative scoring team id per goal.
    # Newer canonical on-ice parquet already includes event_team_id; older versions do not.
    if "event_team_id" in goals_on_ice.columns and goals_on_ice["event_team_id"].notna().any():
        goals_on_ice["scoring_team_id"] = goals_on_ice["event_team_id"]
    else:
        goals_on_ice = goals_on_ice.merge(
            events_df[["event_id", "event_team_id"]],
            on="event_id",
            how="left"
        )
        goals_on_ice["scoring_team_id"] = goals_on_ice["event_team_id"]
    
    # Build player -> team mapping
    player_teams = shifts_df.groupby("player_id")["team_id"].first().to_dict()
    
    # Initialize plus-minus
    plus_minus = {pid: 0 for pid in player_teams.keys()}
    
    for _, goal in goals_on_ice.iterrows():
        scoring_team = goal.get("scoring_team_id")
        
        if pd.isna(scoring_team):
            continue
        
        scoring_team = int(scoring_team)

        # NHL plus/minus excludes power-play goals. Our parsed PBP does not reliably include
        # a strength code, so infer PP from skater advantage at the goal moment:
        # - If scoring team has MORE skaters than defending team => PP goal => exclude
        try:
            home_count = int(goal.get("home_skater_count", 0))
            away_count = int(goal.get("away_skater_count", 0))
        except Exception:
            home_count = 0
            away_count = 0

        # Determine whether scoring side had skater advantage.
        if home_count and away_count:
            scoring_is_home = None
            if "home_team_id" in goal and pd.notna(goal.get("home_team_id")):
                scoring_is_home = int(goal.get("home_team_id")) == scoring_team
            elif "away_team_id" in goal and pd.notna(goal.get("away_team_id")):
                scoring_is_home = int(goal.get("away_team_id")) != scoring_team  # best-effort

            if scoring_is_home is True:
                advantage = home_count - away_count
            elif scoring_is_home is False:
                advantage = away_count - home_count
            else:
                advantage = 0

            # Exclude power-play goals, but DO NOT exclude delayed-penalty goals.
            # Heuristic:
            # - PP goals: scoring team has skater advantage AND their goalie is still in net
            # - Delayed penalty goals: scoring team has skater advantage but goalie is pulled (6v5)
            if advantage > 0:
                scoring_goalie_present = False
                if scoring_is_home is True:
                    scoring_goalie_present = pd.notna(goal.get("home_goalie"))
                elif scoring_is_home is False:
                    scoring_goalie_present = pd.notna(goal.get("away_goalie"))

                if scoring_goalie_present:
                    continue
        
        # Get all skaters on ice (excluding goalies for traditional +/-)
        all_skaters = []
        for side in ["home", "away"]:
            for i in range(1, 7):
                skater = goal.get(f"{side}_skater_{i}")
                if pd.notna(skater):
                    all_skaters.append(int(skater))
        
        # Assign +1 or -1
        for player_id in all_skaters:
            if player_id not in player_teams:
                continue
            
            player_team = player_teams[player_id]
            
            if player_team == scoring_team:
                plus_minus[player_id] = plus_minus.get(player_id, 0) + 1
            else:
                plus_minus[player_id] = plus_minus.get(player_id, 0) - 1
    
    return plus_minus


def validate_plus_minus_vs_boxscore(
    computed_pm: Dict[int, int],
    boxscore_path: Path
) -> Tuple[int, int, int, List[Dict]]:
    """
    Compare computed plus-minus against official boxscore.
    
    Returns: (players_checked, matches, mismatches, details)
    """
    if not boxscore_path.exists():
        return 0, 0, 0, []
    
    with open(boxscore_path) as f:
        boxscore = json.load(f)
    
    # Extract official +/- from boxscore
    official_pm = {}
    
    player_stats = boxscore.get("playerByGameStats", {})
    
    for team_key in ["homeTeam", "awayTeam"]:
        team_data = player_stats.get(team_key, {})
        
        for pos_key in ["forwards", "defense"]:
            players = team_data.get(pos_key, [])
            
            for player in players:
                player_id = player.get("playerId")
                pm = player.get("plusMinus", 0)
                
                if player_id:
                    official_pm[player_id] = pm
    
    if not official_pm:
        return 0, 0, 0, []
    
    # Compare
    mismatches = []
    matches = 0
    
    for player_id, official in official_pm.items():
        computed = computed_pm.get(player_id, 0)
        
        if computed == official:
            matches += 1
        else:
            mismatches.append({
                "player_id": player_id,
                "computed": computed,
                "official": official,
                "difference": computed - official
            })
    
    return len(official_pm), matches, len(mismatches), mismatches


def validate_game(
    staging_dir: Path,
    canonical_dir: Path,
    raw_dir: Path,
    season: str,
    game_id: str
) -> OnIceValidation:
    """Run all validations on a single game."""
    
    result = OnIceValidation(game_id=game_id, season=season)
    
    # Load data
    shifts_path = staging_dir / season / f"{game_id}_shifts.parquet"
    events_path = staging_dir / season / f"{game_id}_events.parquet"
    on_ice_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
    boxscore_path = raw_dir / season / game_id / "boxscore.json"
    
    if not on_ice_path.exists():
        result.errors.append("On-ice data not found. Run build_on_ice.py first.")
        return result
    
    shifts_df = pd.read_parquet(shifts_path)
    events_df = pd.read_parquet(events_path)
    event_on_ice_df = pd.read_parquet(on_ice_path)
    
    # 1. Validate goal scorers on ice
    total_goals, valid_goals, missing = validate_goal_scorer_on_ice(events_df, event_on_ice_df)
    result.total_goals = total_goals
    result.goals_with_scorer_on_ice = valid_goals
    result.goals_missing_scorer = len(missing)
    
    if missing:
        result.warnings.append(f"{len(missing)} goals where scorer not found on ice")
        for m in missing[:3]:  # First 3 details
            result.errors.append(f"Goal {m.get('event_id')}: {m.get('reason')}")
    
    # 2. Validate event team attribution
    _, valid_team, invalid_team = validate_event_team_attribution(events_df, shifts_df)
    result.goals_with_valid_team = valid_team
    result.goals_with_invalid_team = len(invalid_team)
    
    if invalid_team:
        result.errors.append(f"{len(invalid_team)} goals with wrong team attribution")
    
    # 3. Validate 5v5 skater counts
    total_5v5, valid_5v5, invalid_5v5 = validate_5v5_skater_counts(event_on_ice_df)
    result.events_5v5 = total_5v5
    result.events_5v5_valid_counts = valid_5v5
    result.events_5v5_invalid_counts = len(invalid_5v5)
    
    if invalid_5v5:
        result.warnings.append(f"{len(invalid_5v5)} 5v5 events with wrong skater counts")
    
    # 4. Validate plus-minus vs boxscore
    computed_pm = compute_simple_plus_minus(event_on_ice_df, events_df, shifts_df, only_5v5=False)
    checked, matches, mismatches_count, mismatches = validate_plus_minus_vs_boxscore(computed_pm, boxscore_path)
    
    result.players_checked = checked
    result.players_pm_match = matches
    result.players_pm_mismatch = mismatches_count
    
    if mismatches:
        result.max_pm_difference = max(abs(m["difference"]) for m in mismatches)
        
        if result.max_pm_difference > 2:
            result.errors.append(f"Plus-minus mismatch: max difference of {result.max_pm_difference}")
        elif mismatches_count > checked * 0.1:  # >10% mismatch
            result.warnings.append(f"{mismatches_count}/{checked} players with +/- mismatch")
    
    # Overall pass/fail
    result.all_passed = (
        len(result.errors) == 0 and
        result.goals_missing_scorer == 0 and
        result.goals_with_invalid_team == 0 and
        result.max_pm_difference <= 2
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate on-ice assignments")
    parser.add_argument("--game", type=str, help="Specific game to validate")
    parser.add_argument("--season", type=str, default=None, help="Only validate a specific season (e.g., 20252026)")
    parser.add_argument("--strict", action="store_true", help="Fail on any warning")
    
    args = parser.parse_args()
    
    staging_dir = Path(__file__).parent.parent / "staging"
    canonical_dir = Path(__file__).parent.parent / "canonical"
    raw_dir = Path(__file__).parent.parent / "raw"
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NHL Data Pipeline - Validate On-Ice Assignments")
    print("=" * 60)
    
    # Find games to validate
    if args.game:
        games = []
        for season_dir in canonical_dir.glob("*"):
            if (season_dir / f"{args.game}_event_on_ice.parquet").exists():
                games.append((season_dir.name, args.game))
                break
        
        if not games:
            print(f"Game {args.game} not found in canonical data.")
            print("Run build_on_ice.py first.")
            return
    else:
        games = []
        if args.season:
            onice_glob = canonical_dir / args.season
            onice_files = onice_glob.glob("*_event_on_ice.parquet")
        else:
            onice_files = canonical_dir.glob("*/*_event_on_ice.parquet")

        for on_ice_file in onice_files:
            game_id = on_ice_file.stem.replace("_event_on_ice", "")
            season = on_ice_file.parent.name
            games.append((season, game_id))
    
    if not games:
        print("No games found. Run build_on_ice.py first.")
        return
    
    print(f"Validating {len(games)} games...")
    
    results = []
    
    for season, game_id in sorted(games):
        print(f"\n{season}/{game_id}...")
        
        result = validate_game(staging_dir, canonical_dir, raw_dir, season, game_id)
        results.append(result)
        
        # Print summary
        if result.all_passed:
            print(f"  OK PASSED")
            print(f"    Goals: {result.goals_with_scorer_on_ice}/{result.total_goals} scorers on ice")
            print(f"    +/-: {result.players_pm_match}/{result.players_checked} match boxscore")
        else:
            print(f"  FAIL FAILED")
            for err in result.errors:
                print(f"    ERROR: {err}")
            for warn in result.warnings:
                print(f"    WARN: {warn}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r.all_passed)
    total = len(results)
    
    print(f"Games passed: {passed}/{total} ({100*passed/total:.1f}%)")
    
    total_goals = sum(r.total_goals for r in results)
    valid_goals = sum(r.goals_with_scorer_on_ice for r in results)
    print(f"Goals with scorer on ice: {valid_goals}/{total_goals} ({100*valid_goals/total_goals:.1f}%)")
    
    total_checked = sum(r.players_checked for r in results)
    total_match = sum(r.players_pm_match for r in results)
    if total_checked > 0:
        print(f"+/- matches boxscore: {total_match}/{total_checked} ({100*total_match/total_checked:.1f}%)")
    
    # Save results (merge into a single cross-season file so downstream steps can enforce Gate 2)
    output_path = data_dir / "on_ice_validation.json"
    merged: dict[str, dict] = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for r in existing:
                    if isinstance(r, dict):
                        key = f"{r.get('season')}::{r.get('game_id')}"
                        merged[key] = r
        except Exception:
            # If the file is corrupt/unreadable, fall back to overwriting with current run.
            merged = {}

    for r in results:
        d = asdict(r)
        key = f"{d.get('season')}::{d.get('game_id')}"
        merged[key] = d

    merged_list = list(merged.values())
    merged_list.sort(key=lambda x: (str(x.get("season", "")), str(x.get("game_id", ""))))
    output_path.write_text(json.dumps(merged_list, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path} ({len(merged_list)} total game results)")
    
    # Exit code
    if passed < total:
        if args.strict:
            print("\nFAIL STRICT MODE: Some games failed validation")
            exit(1)
        else:
            print("\nWARN: Some games failed validation. Review before computing APM.")
    else:
        print("\nOK All games passed. Ready for APM computation.")


if __name__ == "__main__":
    main()

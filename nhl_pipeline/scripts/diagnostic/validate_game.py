#!/usr/bin/env python3
"""
Validate parsed NHL data with concrete integrity tests.

These tests MUST pass before any modeling happens.
Each test has clear pass/fail criteria with specific tolerances.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class GameValidation:
    """All validation results for a single game."""
    game_id: str
    season: str
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)
    
    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)


def time_to_seconds(time_str: str) -> int:
    """Convert "MM:SS" to seconds."""
    if not time_str:
        return 0
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
    except:
        return 0


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_shift_duration_sum(
    shifts_df: pd.DataFrame, 
    boxscore_data: Dict,
    tolerance_seconds: int = 120  # 2 minute tolerance
) -> ValidationResult:
    """
    TEST 1: Total shift duration approximately matches boxscore TOI.
    
    For each player, sum of shift durations should be within tolerance
    of the reported boxscore TOI.
    
    This catches:
    - Missing shifts
    - Duplicate shifts
    - Incorrectly parsed durations
    """
    test_name = "shift_duration_vs_boxscore"
    
    try:
        # Get boxscore TOI per player
        boxscore_toi = {}
        
        # Parse boxscore structure
        player_by_game_stats = boxscore_data.get("playerByGameStats", {})
        
        for team_key in ["homeTeam", "awayTeam"]:
            team_data = player_by_game_stats.get(team_key, {})
            
            # Forwards and defense
            for pos_key in ["forwards", "defense"]:
                players = team_data.get(pos_key, [])
                for player in players:
                    player_id = player.get("playerId")
                    toi_str = player.get("toi", "0:00")
                    if player_id:
                        boxscore_toi[player_id] = time_to_seconds(toi_str)
        
        if not boxscore_toi:
            return ValidationResult(
                test_name=test_name,
                passed=False,
                message="Could not extract TOI from boxscore",
                details={"boxscore_keys": list(boxscore_data.keys())}
            )
        
        # Calculate shift-based TOI per player
        shift_toi = shifts_df.groupby("player_id")["duration_seconds"].sum().to_dict()
        
        # Compare
        mismatches = []
        for player_id, box_toi in boxscore_toi.items():
            shift_sum = shift_toi.get(player_id, 0)
            diff = abs(box_toi - shift_sum)
            
            if diff > tolerance_seconds:
                mismatches.append({
                    "player_id": player_id,
                    "boxscore_toi": box_toi,
                    "shift_toi": shift_sum,
                    "diff": diff
                })
        
        if mismatches:
            return ValidationResult(
                test_name=test_name,
                passed=False,
                message=f"{len(mismatches)} players with TOI mismatch > {tolerance_seconds}s",
                details={
                    "tolerance_seconds": tolerance_seconds,
                    "mismatches": mismatches[:10],  # First 10
                    "total_mismatches": len(mismatches)
                }
            )
        
        return ValidationResult(
            test_name=test_name,
            passed=True,
            message=f"All {len(boxscore_toi)} players within {tolerance_seconds}s tolerance",
            details={
                "players_checked": len(boxscore_toi),
                "tolerance_seconds": tolerance_seconds
            }
        )
        
    except Exception as e:
        return ValidationResult(
            test_name=test_name,
            passed=False,
            message=f"Test error: {str(e)}",
            details={"error": str(e)}
        )


def test_no_overlapping_shifts(shifts_df: pd.DataFrame) -> ValidationResult:
    """
    TEST 2: No player has overlapping shifts in the same period.
    
    For each player in each period, shifts should not overlap in time.
    A small overlap (1-2 seconds) is tolerated due to timing granularity.
    
    This catches:
    - Duplicate shift records
    - Data corruption
    - Parsing errors
    """
    test_name = "no_overlapping_shifts"
    
    try:
        overlap_tolerance = 2  # seconds
        overlaps = []
        
        for (player_id, period), group in shifts_df.groupby(["player_id", "period"]):
            # Sort by start time
            sorted_shifts = group.sort_values("start_seconds")
            
            prev_end = -1
            for _, shift in sorted_shifts.iterrows():
                start = shift["start_seconds"]
                end = shift["end_seconds"]
                
                if start < prev_end - overlap_tolerance:
                    overlaps.append({
                        "player_id": player_id,
                        "period": period,
                        "prev_end": prev_end,
                        "next_start": start,
                        "overlap": prev_end - start
                    })
                
                prev_end = max(prev_end, end)
        
        if overlaps:
            return ValidationResult(
                test_name=test_name,
                passed=False,
                message=f"{len(overlaps)} overlapping shifts detected",
                details={
                    "overlaps": overlaps[:10],
                    "total_overlaps": len(overlaps)
                }
            )
        
        return ValidationResult(
            test_name=test_name,
            passed=True,
            message="No overlapping shifts detected",
            details={"shifts_checked": len(shifts_df)}
        )
        
    except Exception as e:
        return ValidationResult(
            test_name=test_name,
            passed=False,
            message=f"Test error: {str(e)}",
            details={"error": str(e)}
        )


def test_goals_have_on_ice_players(
    events_df: pd.DataFrame,
    shifts_df: pd.DataFrame
) -> ValidationResult:
    """
    TEST 3: Every goal scorer was on the ice when the goal occurred.
    
    For each goal event, the scoring player should have an active shift
    at that game time.
    
    This catches:
    - Shift/event time misalignment
    - Wrong player attribution
    - Missing shift data
    """
    test_name = "goals_have_on_ice_players"
    
    try:
        goals = events_df[events_df["event_type"] == "GOAL"].copy()
        
        if goals.empty:
            return ValidationResult(
                test_name=test_name,
                passed=True,
                message="No goals in this game to validate",
                details={}
            )
        
        missing_on_ice = []
        
        for _, goal in goals.iterrows():
            scorer_id = goal.get("player_1_id")
            period = goal.get("period")
            goal_time = goal.get("period_seconds")
            
            if pd.isna(scorer_id):
                continue
            
            scorer_id = int(scorer_id)
            
            # Find shifts for this player in this period that contain the goal time
            player_shifts = shifts_df[
                (shifts_df["player_id"] == scorer_id) &
                (shifts_df["period"] == period)
            ]
            
            # Check if any shift contains the goal time
            on_ice = False
            for _, shift in player_shifts.iterrows():
                # Allow 2 second tolerance on shift boundaries
                if shift["start_seconds"] - 2 <= goal_time <= shift["end_seconds"] + 2:
                    on_ice = True
                    break
            
            if not on_ice:
                missing_on_ice.append({
                    "event_id": goal.get("event_id"),
                    "scorer_id": scorer_id,
                    "period": period,
                    "goal_time": goal_time,
                    "player_shifts_in_period": len(player_shifts)
                })
        
        if missing_on_ice:
            return ValidationResult(
                test_name=test_name,
                passed=False,
                message=f"{len(missing_on_ice)} goals where scorer not on ice",
                details={
                    "missing": missing_on_ice,
                    "total_goals": len(goals)
                }
            )
        
        return ValidationResult(
            test_name=test_name,
            passed=True,
            message=f"All {len(goals)} goal scorers confirmed on ice",
            details={"goals_checked": len(goals)}
        )
        
    except Exception as e:
        return ValidationResult(
            test_name=test_name,
            passed=False,
            message=f"Test error: {str(e)}",
            details={"error": str(e)}
        )


def test_exactly_two_teams(shifts_df: pd.DataFrame) -> ValidationResult:
    """
    TEST 4: Exactly two teams in the shift data.
    
    Basic sanity check.
    """
    test_name = "exactly_two_teams"
    
    unique_teams = shifts_df["team_id"].unique()
    
    if len(unique_teams) == 2:
        return ValidationResult(
            test_name=test_name,
            passed=True,
            message=f"Exactly 2 teams found: {list(unique_teams)}",
            details={"team_ids": list(unique_teams)}
        )
    else:
        return ValidationResult(
            test_name=test_name,
            passed=False,
            message=f"Expected 2 teams, found {len(unique_teams)}",
            details={"team_ids": list(unique_teams)}
        )


def test_reasonable_shift_counts(shifts_df: pd.DataFrame) -> ValidationResult:
    """
    TEST 5: Each team has a reasonable number of shifts.
    
    A typical NHL game has 200-400 shifts per team (regular time).
    This catches completely missing data.
    """
    test_name = "reasonable_shift_counts"
    
    min_shifts_per_team = 100
    max_shifts_per_team = 600
    
    shifts_per_team = shifts_df.groupby("team_id").size()
    
    issues = []
    for team_id, count in shifts_per_team.items():
        if count < min_shifts_per_team:
            issues.append(f"Team {team_id}: only {count} shifts (min: {min_shifts_per_team})")
        elif count > max_shifts_per_team:
            issues.append(f"Team {team_id}: {count} shifts (max: {max_shifts_per_team})")
    
    if issues:
        return ValidationResult(
            test_name=test_name,
            passed=False,
            message="; ".join(issues),
            details={"shifts_per_team": shifts_per_team.to_dict()}
        )
    
    return ValidationResult(
        test_name=test_name,
        passed=True,
        message=f"Shift counts within expected range",
        details={"shifts_per_team": shifts_per_team.to_dict()}
    )


def test_event_types_present(events_df: pd.DataFrame) -> ValidationResult:
    """
    TEST 6: Essential event types are present.
    
    A valid game should have shots, goals (usually), faceoffs, etc.
    """
    test_name = "essential_event_types"
    
    required_types = {"SHOT", "FACEOFF", "PERIOD_START", "PERIOD_END"}
    expected_types = {"GOAL", "HIT", "BLOCKED_SHOT", "MISSED_SHOT"}
    
    present_types = set(events_df["event_type"].unique())
    
    missing_required = required_types - present_types
    missing_expected = expected_types - present_types
    
    if missing_required:
        return ValidationResult(
            test_name=test_name,
            passed=False,
            message=f"Missing required event types: {missing_required}",
            details={
                "present": list(present_types),
                "missing_required": list(missing_required)
            }
        )
    
    return ValidationResult(
        test_name=test_name,
        passed=True,
        message=f"All required event types present",
        details={
            "present": list(present_types),
            "missing_expected": list(missing_expected) if missing_expected else []
        }
    )


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def validate_game(
    staging_dir: Path,
    raw_dir: Path,
    season: str,
    game_id: str
) -> GameValidation:
    """Run all validation tests on a single game."""
    
    validation = GameValidation(game_id=game_id, season=season)
    
    # Load data
    shifts_path = staging_dir / season / f"{game_id}_shifts.parquet"
    events_path = staging_dir / season / f"{game_id}_events.parquet"
    boxscore_path = raw_dir / season / game_id / "boxscore.json"
    
    if not shifts_path.exists():
        validation.results.append(ValidationResult(
            test_name="data_exists",
            passed=False,
            message=f"Shifts file not found: {shifts_path}"
        ))
        return validation
    
    if not events_path.exists():
        validation.results.append(ValidationResult(
            test_name="data_exists",
            passed=False,
            message=f"Events file not found: {events_path}"
        ))
        return validation
    
    shifts_df = pd.read_parquet(shifts_path)
    events_df = pd.read_parquet(events_path)
    
    with open(boxscore_path) as f:
        boxscore_data = json.load(f)
    
    # Run tests
    validation.results.append(test_exactly_two_teams(shifts_df))
    validation.results.append(test_reasonable_shift_counts(shifts_df))
    validation.results.append(test_no_overlapping_shifts(shifts_df))
    validation.results.append(test_shift_duration_sum(shifts_df, boxscore_data))
    validation.results.append(test_event_types_present(events_df))
    validation.results.append(test_goals_have_on_ice_players(events_df, shifts_df))
    
    return validation


def main():
    """Run validation on all parsed games."""
    raw_dir = Path(__file__).parent.parent.parent / "raw"
    staging_dir = Path(__file__).parent.parent.parent / "staging"
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NHL Data Pipeline - Validation")
    print("=" * 60)
    
    # Import history tracker
    from validation_history import ValidationHistory, create_validation_run
    
    history_path = data_dir / "validation_history.parquet"
    history = ValidationHistory(history_path)
    
    # Find all staged games
    staged_games = []
    for shifts_file in staging_dir.glob("*/*_shifts.parquet"):
        game_id = shifts_file.stem.replace("_shifts", "")
        season = shifts_file.parent.name
        staged_games.append((season, game_id))
    
    print(f"Found {len(staged_games)} games to validate")
    
    all_validations = []
    
    for season, game_id in sorted(staged_games):
        print(f"\nValidating {season}/{game_id}...")
        
        validation = validate_game(staging_dir, raw_dir, season, game_id)
        all_validations.append(validation)
        
        # Load dataframes for additional metrics
        events_df = None
        events_path = staging_dir / season / f"{game_id}_events.parquet"
        if events_path.exists():
            events_df = pd.read_parquet(events_path)
        
        # Create and store validation run
        results_dicts = [
            {"test_name": r.test_name, "passed": r.passed, "message": r.message, "details": r.details}
            for r in validation.results
        ]
        
        run = create_validation_run(
            season=season,
            game_id=game_id,
            validation_results=results_dicts,
            events_df=events_df,
            raw_path=str(raw_dir / season / game_id),
        )
        history.add_run(run)
        
        for result in validation.results:
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.test_name}: {result.message}")
    
    # Save history
    history.save()
    print(f"\n✓ Validation history saved to {history_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total_passed = sum(v.all_passed for v in all_validations)
    total_games = len(all_validations)
    
    print(f"Games fully validated: {total_passed}/{total_games}")
    
    for v in all_validations:
        status = "✓" if v.all_passed else "✗"
        print(f"  {status} {v.season}/{v.game_id}: {v.pass_count}/{len(v.results)} tests passed")
        
        # Show failures
        for r in v.results:
            if not r.passed:
                print(f"      FAIL: {r.test_name} - {r.message}")
    
    # Return validation data for further analysis
    return all_validations


if __name__ == "__main__":
    main()

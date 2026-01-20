#!/usr/bin/env python3
"""
Validation History - Track validation results over time.

Stores every validation run so you can:
- See trends in data quality
- Identify when problems started
- Compare pass rates across seasons
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import pandas as pd


@dataclass
class ValidationRun:
    """A single validation run for one game."""
    run_id: str
    run_timestamp: str
    season: str
    game_id: str
    
    # Test results
    test_two_teams: bool = False
    test_shift_counts: bool = False
    test_no_overlaps: bool = False
    test_toi_match: bool = False
    test_event_types: bool = False
    test_goals_on_ice: bool = False
    
    # Metrics
    toi_diff_avg: float = 0.0
    toi_diff_max: float = 0.0
    overlap_count: int = 0
    missing_goals_count: int = 0
    home_shifts: int = 0
    away_shifts: int = 0
    total_events: int = 0
    total_goals: int = 0
    total_shots: int = 0
    
    # Overall
    tests_passed: int = 0
    tests_total: int = 6
    all_passed: bool = False
    
    # Debug info
    error_message: Optional[str] = None
    raw_json_path: Optional[str] = None


class ValidationHistory:
    """Track validation history across pipeline runs."""
    
    def __init__(self, history_path: Path):
        self.history_path = history_path
        self.runs: List[ValidationRun] = []
        self._load()
    
    def _load(self):
        """Load existing history from disk."""
        if self.history_path.exists():
            df = pd.read_parquet(self.history_path)
            for _, row in df.iterrows():
                run = ValidationRun(
                    run_id=row["run_id"],
                    run_timestamp=row["run_timestamp"],
                    season=row["season"],
                    game_id=row["game_id"],
                    test_two_teams=row.get("test_two_teams", False),
                    test_shift_counts=row.get("test_shift_counts", False),
                    test_no_overlaps=row.get("test_no_overlaps", False),
                    test_toi_match=row.get("test_toi_match", False),
                    test_event_types=row.get("test_event_types", False),
                    test_goals_on_ice=row.get("test_goals_on_ice", False),
                    toi_diff_avg=row.get("toi_diff_avg", 0.0),
                    toi_diff_max=row.get("toi_diff_max", 0.0),
                    overlap_count=row.get("overlap_count", 0),
                    missing_goals_count=row.get("missing_goals_count", 0),
                    home_shifts=row.get("home_shifts", 0),
                    away_shifts=row.get("away_shifts", 0),
                    total_events=row.get("total_events", 0),
                    total_goals=row.get("total_goals", 0),
                    total_shots=row.get("total_shots", 0),
                    tests_passed=row.get("tests_passed", 0),
                    tests_total=row.get("tests_total", 6),
                    all_passed=row.get("all_passed", False),
                    error_message=row.get("error_message"),
                    raw_json_path=row.get("raw_json_path"),
                )
                self.runs.append(run)
    
    def _save(self):
        """Save history to disk."""
        if not self.runs:
            return
        
        rows = [asdict(run) for run in self.runs]
        df = pd.DataFrame(rows)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.history_path, index=False)
    
    def add_run(self, run: ValidationRun):
        """Add a new validation run."""
        self.runs.append(run)
    
    def save(self):
        """Persist history to disk."""
        self._save()
    
    def get_latest_by_game(self) -> pd.DataFrame:
        """Get most recent validation for each game."""
        if not self.runs:
            return pd.DataFrame()
        
        df = pd.DataFrame([asdict(r) for r in self.runs])
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
        
        # Get latest run per game
        latest = df.sort_values("run_timestamp").groupby(["season", "game_id"]).last().reset_index()
        return latest
    
    def get_summary_by_season(self) -> pd.DataFrame:
        """Get aggregated stats by season."""
        latest = self.get_latest_by_game()
        if latest.empty:
            return latest
        
        summary = latest.groupby("season").agg({
            "game_id": "count",
            "all_passed": ["sum", "mean"],
            "toi_diff_avg": "mean",
            "overlap_count": "sum",
            "missing_goals_count": "sum",
            "home_shifts": "mean",
            "away_shifts": "mean",
        }).reset_index()
        
        summary.columns = [
            "season", "games", "passed", "pass_rate",
            "avg_toi_diff", "total_overlaps", "total_missing_goals",
            "avg_home_shifts", "avg_away_shifts"
        ]
        
        summary["pass_rate"] = (summary["pass_rate"] * 100).round(1)
        summary["failed"] = summary["games"] - summary["passed"]
        
        return summary.sort_values("season", ascending=False)
    
    def get_failed_games(self) -> pd.DataFrame:
        """Get all games that failed validation."""
        latest = self.get_latest_by_game()
        if latest.empty:
            return latest
        
        failed = latest[~latest["all_passed"]]
        return failed[[
            "season", "game_id", "tests_passed", "tests_total",
            "toi_diff_avg", "overlap_count", "missing_goals_count",
            "error_message"
        ]].sort_values(["season", "game_id"], ascending=[False, True])
    
    def get_trends(self, metric: str = "pass_rate", window: int = 10) -> pd.DataFrame:
        """Get rolling trends for a metric."""
        if not self.runs:
            return pd.DataFrame()
        
        df = pd.DataFrame([asdict(r) for r in self.runs])
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
        df = df.sort_values("run_timestamp")
        
        # Calculate rolling average
        if metric == "pass_rate":
            df["metric"] = df["all_passed"].astype(float) * 100
        elif metric in df.columns:
            df["metric"] = df[metric]
        else:
            return pd.DataFrame()
        
        df["rolling_avg"] = df["metric"].rolling(window=window, min_periods=1).mean()
        
        return df[["run_timestamp", "season", "game_id", "metric", "rolling_avg"]]


def create_validation_run(
    season: str,
    game_id: str,
    validation_results: List[Dict],
    shifts_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
    raw_path: Optional[str] = None,
) -> ValidationRun:
    """Create a ValidationRun from validation results."""
    
    run = ValidationRun(
        run_id=str(uuid.uuid4()),
        run_timestamp=datetime.now().isoformat(),
        season=season,
        game_id=game_id,
        raw_json_path=raw_path,
    )
    
    # Parse validation results
    for result in validation_results:
        name = result.get("test_name", "")
        passed = result.get("passed", False)
        details = result.get("details", {})
        
        if name == "exactly_two_teams":
            run.test_two_teams = passed
        elif name == "reasonable_shift_counts":
            run.test_shift_counts = passed
            run.home_shifts = details.get("shifts_per_team", {}).get(list(details.get("shifts_per_team", {}).keys())[0], 0) if details.get("shifts_per_team") else 0
            run.away_shifts = details.get("shifts_per_team", {}).get(list(details.get("shifts_per_team", {}).keys())[1], 0) if details.get("shifts_per_team") and len(details.get("shifts_per_team", {})) > 1 else 0
        elif name == "no_overlapping_shifts":
            run.test_no_overlaps = passed
            run.overlap_count = details.get("total_overlaps", 0)
        elif name == "shift_duration_vs_boxscore":
            run.test_toi_match = passed
            # Extract TOI diff from mismatches if present
            mismatches = details.get("mismatches", [])
            if mismatches:
                diffs = [m.get("diff", 0) for m in mismatches]
                run.toi_diff_avg = sum(diffs) / len(diffs) if diffs else 0
                run.toi_diff_max = max(diffs) if diffs else 0
        elif name == "essential_event_types":
            run.test_event_types = passed
        elif name == "goals_have_on_ice_players":
            run.test_goals_on_ice = passed
            run.missing_goals_count = details.get("missing", 0) if not passed else 0
    
    # Count passes
    tests = [
        run.test_two_teams, run.test_shift_counts, run.test_no_overlaps,
        run.test_toi_match, run.test_event_types, run.test_goals_on_ice
    ]
    run.tests_passed = sum(tests)
    run.tests_total = len(tests)
    run.all_passed = all(tests)
    
    # Add event stats if available
    if events_df is not None and not events_df.empty:
        run.total_events = len(events_df)
        run.total_goals = len(events_df[events_df["event_type"] == "GOAL"])
        run.total_shots = len(events_df[events_df["event_type"] == "SHOT"])
    
    return run


def main():
    """Demo the validation history."""
    
    history_path = Path(__file__).parent.parent / "data" / "validation_history.parquet"
    history = ValidationHistory(history_path)
    
    print("=" * 70)
    print("VALIDATION HISTORY")
    print("=" * 70)
    
    if not history.runs:
        print("\nNo validation history yet. Run the pipeline first.")
        return
    
    # Summary by season
    print("\n📊 SUMMARY BY SEASON")
    print("-" * 60)
    
    summary = history.get_summary_by_season()
    if not summary.empty:
        print(f"{'Season':<12} {'Games':>6} {'Passed':>8} {'Failed':>8} {'Pass%':>8} {'Avg TOI Diff':>12}")
        print("-" * 60)
        for _, row in summary.iterrows():
            status = "✓" if row["pass_rate"] == 100 else "⚠️" if row["pass_rate"] >= 80 else "✗"
            print(f"{row['season']:<12} {int(row['games']):>6} {int(row['passed']):>8} {int(row['failed']):>8} {row['pass_rate']:>7.1f}% {row['avg_toi_diff']:>11.1f}s {status}")
    
    # Failed games
    failed = history.get_failed_games()
    if not failed.empty:
        print(f"\n❌ FAILED GAMES ({len(failed)} total)")
        print("-" * 60)
        for _, row in failed.head(10).iterrows():
            reasons = []
            if row["toi_diff_avg"] > 120:
                reasons.append(f"TOI:{row['toi_diff_avg']:.0f}s")
            if row["overlap_count"] > 0:
                reasons.append(f"overlaps:{row['overlap_count']}")
            if row["missing_goals_count"] > 0:
                reasons.append(f"missing_goals:{row['missing_goals_count']}")
            
            reason_str = ", ".join(reasons) if reasons else "unknown"
            print(f"  {row['season']}/{row['game_id']}: {row['tests_passed']}/{row['tests_total']} ({reason_str})")
        
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    # Overall stats
    latest = history.get_latest_by_game()
    if not latest.empty:
        print(f"\n📈 OVERALL STATS")
        print("-" * 60)
        print(f"  Total games tracked:  {len(latest)}")
        print(f"  Total passed:         {latest['all_passed'].sum()}")
        print(f"  Overall pass rate:    {100 * latest['all_passed'].mean():.1f}%")
        print(f"  Avg TOI difference:   {latest['toi_diff_avg'].mean():.1f}s")
        print(f"  Total overlaps:       {latest['overlap_count'].sum()}")


if __name__ == "__main__":
    main()

"""
RAPM Pipeline Validation Suite

Run this script BEFORE and AFTER the full pipeline to ensure data quality.

Usage:
    python validate_rapm_pipeline.py --pre   # Run before pipeline
    python validate_rapm_pipeline.py --post  # Run after pipeline
    python validate_rapm_pipeline.py --all   # Run all checks
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pandas as pd


class RAPMValidator:
    """Validates RAPM pipeline data quality."""
    
    def __init__(self, db_path: str = "nhl_canonical.duckdb"):
        self.db_path = db_path
        self.staging_dir = Path("staging")
        self.canonical_dir = Path("canonical")
        self.data_dir = Path("data")
        self.results: List[Dict] = []
        
    def log(self, check_name: str, passed: bool, message: str, severity: str = "ERROR"):
        """Log a validation result."""
        status = "PASS" if passed else f"FAIL ({severity})"
        self.results.append({
            "check": check_name,
            "passed": passed,
            "message": message,
            "severity": severity if not passed else None
        })
        print(f"  [{status}] {check_name}: {message}")
        
    def run_pre_checks(self) -> bool:
        """Run checks before pipeline execution."""
        print("\n" + "=" * 70)
        print("PRE-PIPELINE VALIDATION CHECKS")
        print("=" * 70 + "\n")
        
        self._check_staging_data_exists()
        self._check_canonical_data_exists()
        self._check_gate2_validation()
        self._check_sample_game_data_quality()
        
        return all(r["passed"] for r in self.results)
    
    def run_post_checks(self) -> bool:
        """Run checks after pipeline execution."""
        print("\n" + "=" * 70)
        print("POST-PIPELINE VALIDATION CHECKS")
        print("=" * 70 + "\n")
        
        self._check_games_count_consistency()
        self._check_events_count_reasonable()
        self._check_elite_player_rankings()
        self._check_rapm_value_distributions()
        self._check_all_metrics_present()
        
        return all(r["passed"] for r in self.results)
    
    # =========================================================================
    # PRE-PIPELINE CHECKS
    # =========================================================================
    
    def _check_staging_data_exists(self):
        """Verify staging data exists for all seasons."""
        seasons = list(self.staging_dir.glob("20*"))
        if not seasons:
            self.log("staging_data", False, "No season directories in staging/", "CRITICAL")
            return
        
        season_counts = {}
        for season_dir in seasons:
            event_files = list(season_dir.glob("*_events.parquet"))
            season_counts[season_dir.name] = len(event_files)
        
        total = sum(season_counts.values())
        self.log(
            "staging_data", 
            total > 0, 
            f"Found {total} games across {len(seasons)} seasons: {season_counts}"
        )
    
    def _check_canonical_data_exists(self):
        """Verify canonical (on-ice) data exists."""
        seasons = list(self.canonical_dir.glob("20*"))
        if not seasons:
            self.log("canonical_data", False, "No season directories in canonical/", "CRITICAL")
            return
        
        season_counts = {}
        for season_dir in seasons:
            onice_files = list(season_dir.glob("*_event_on_ice.parquet"))
            season_counts[season_dir.name] = len(onice_files)
        
        total = sum(season_counts.values())
        self.log(
            "canonical_data", 
            total > 0, 
            f"Found {total} on-ice files across {len(seasons)} seasons"
        )
    
    def _check_gate2_validation(self):
        """Check Gate 2 validation file exists and has passed games."""
        val_path = self.data_dir / "on_ice_validation.json"
        if not val_path.exists():
            self.log("gate2_validation", False, "on_ice_validation.json not found", "CRITICAL")
            return
        
        try:
            data = json.loads(val_path.read_text())
            passed = [r for r in data if r.get("all_passed")]
            by_season = {}
            for r in passed:
                s = str(r.get("season"))
                by_season[s] = by_season.get(s, 0) + 1
            
            self.log(
                "gate2_validation",
                len(passed) > 0,
                f"{len(passed)} games passed Gate 2: {by_season}"
            )
        except Exception as e:
            self.log("gate2_validation", False, f"Failed to parse: {e}", "CRITICAL")
    
    def _check_sample_game_data_quality(self):
        """Check sample game has expected columns and data."""
        seasons = list(self.staging_dir.glob("20*"))
        if not seasons:
            return
        
        # Get latest season
        latest = sorted(seasons)[-1]
        event_files = list(latest.glob("*_events.parquet"))
        if not event_files:
            self.log("sample_game", False, f"No events in {latest.name}", "WARNING")
            return
        
        try:
            df = pd.read_parquet(event_files[0])
            required_cols = ["event_id", "event_type", "player_1_id", "player_2_id", "player_3_id"]
            missing = [c for c in required_cols if c not in df.columns]
            
            if missing:
                self.log("sample_game", False, f"Missing columns: {missing}", "CRITICAL")
                return
            
            # Check GOAL events have assists
            goals = df[df["event_type"] == "GOAL"]
            with_a1 = goals["player_2_id"].notna().sum()
            
            self.log(
                "sample_game",
                len(goals) > 0 and with_a1 > 0,
                f"Sample game: {len(goals)} goals, {with_a1} with primary assists"
            )
        except Exception as e:
            self.log("sample_game", False, f"Error reading: {e}", "ERROR")
    
    # =========================================================================
    # POST-PIPELINE CHECKS
    # =========================================================================
    
    def _check_games_count_consistency(self):
        """Verify all metrics use same games_count per season."""
        con = duckdb.connect(self.db_path, read_only=True)
        
        try:
            result = con.execute("""
                SELECT season,
                       COUNT(DISTINCT games_count) as distinct_counts,
                       MIN(games_count) as min_games,
                       MAX(games_count) as max_games
                FROM apm_results
                GROUP BY season
                ORDER BY season DESC
            """).fetchall()
            
            inconsistent = [(s, mn, mx) for s, dc, mn, mx in result if dc > 1]
            
            if inconsistent:
                msg = ", ".join([f"{s}: {mn}-{mx} games" for s, mn, mx in inconsistent])
                self.log(
                    "games_consistency",
                    False,
                    f"Inconsistent games_count: {msg}",
                    "ERROR"
                )
            else:
                games_by_season = {s: mx for s, dc, mn, mx in result}
                self.log(
                    "games_consistency",
                    True,
                    f"Consistent across all metrics: {games_by_season}"
                )
        finally:
            con.close()
    
    def _check_events_count_reasonable(self):
        """Verify event counts match expected ranges."""
        con = duckdb.connect(self.db_path, read_only=True)
        
        try:
            result = con.execute("""
                SELECT season, 
                       MAX(games_count) as games,
                       MAX(CASE WHEN metric_name = 'corsi_rapm_5v5' THEN events_count END) as corsi_events,
                       MAX(CASE WHEN metric_name = 'goals_rapm_5v5' THEN events_count END) as goals_events,
                       MAX(CASE WHEN metric_name = 'primary_assist_rapm_5v5' THEN events_count END) as a1_events
                FROM apm_results
                GROUP BY season
                ORDER BY season DESC
            """).fetchall()
            
            issues = []
            for season, games, corsi, goals, a1 in result:
                if games is None:
                    continue
                    
                # Expected: ~100 corsi per game, ~4 5v5 goals per game, ~3 5v5 assists per game
                expected_corsi = games * 100
                expected_goals = games * 4  # ~64% of goals at 5v5
                expected_a1 = games * 3
                
                if corsi is not None and corsi < expected_corsi * 0.5:
                    issues.append(f"{season}: corsi={corsi} (expected ~{expected_corsi})")
                if goals is not None and goals < expected_goals * 0.5:
                    issues.append(f"{season}: goals={goals} (expected ~{expected_goals})")
                if a1 is not None and a1 < expected_a1 * 0.5:
                    issues.append(f"{season}: a1={a1} (expected ~{expected_a1})")
            
            if issues:
                self.log("events_reasonable", False, f"Low event counts: {issues[:5]}", "WARNING")
            else:
                self.log("events_reasonable", True, "Event counts within expected ranges")
        finally:
            con.close()
    
    def _check_elite_player_rankings(self):
        """Verify elite players rank reasonably in assist metrics."""
        con = duckdb.connect(self.db_path, read_only=True)
        
        # Known elite playmakers
        elite = {
            8478402: "McDavid",
            8477492: "MacKinnon", 
            8476453: "Kucherov",
            8477934: "Draisaitl"
        }
        
        try:
            result = con.execute("""
                WITH ranked AS (
                    SELECT player_id, value,
                           ROW_NUMBER() OVER (ORDER BY value DESC) as rank
                    FROM apm_results
                    WHERE metric_name = 'primary_assist_rapm_5v5'
                      AND season = (SELECT MAX(season) FROM apm_results)
                )
                SELECT player_id, rank FROM ranked
                WHERE player_id IN (8478402, 8477492, 8476453, 8477934)
            """).fetchall()
            
            if not result:
                self.log("elite_rankings", False, "No elite players found in results", "WARNING")
                return
            
            rankings = {pid: rank for pid, rank in result}
            top_50_count = sum(1 for r in rankings.values() if r <= 50)
            
            details = ", ".join([f"{elite.get(pid, pid)}:#{r}" for pid, r in sorted(rankings.items(), key=lambda x: x[1])])
            
            self.log(
                "elite_rankings",
                top_50_count >= 2,
                f"Elite playmakers: {details} ({top_50_count}/4 in top 50)"
            )
        finally:
            con.close()
    
    def _check_rapm_value_distributions(self):
        """Check RAPM values have reasonable distributions."""
        con = duckdb.connect(self.db_path, read_only=True)
        
        try:
            result = con.execute("""
                SELECT metric_name,
                       AVG(value) as mean_val,
                       STDDEV(value) as std_val,
                       MIN(value) as min_val,
                       MAX(value) as max_val
                FROM apm_results
                WHERE season = (SELECT MAX(season) FROM apm_results)
                GROUP BY metric_name
            """).fetchall()
            
            issues = []
            for metric, mean, std, min_v, max_v in result:
                # Check mean is near zero (balanced)
                if abs(mean) > 0.1:
                    issues.append(f"{metric}: mean={mean:.3f} (should be ~0)")
                # Check reasonable spread
                if std is not None and std < 0.01:
                    issues.append(f"{metric}: std={std:.4f} (too narrow)")
            
            if issues:
                self.log("value_distribution", False, f"Distribution issues: {issues[:5]}", "WARNING")
            else:
                self.log("value_distribution", True, "All metrics have reasonable distributions")
        finally:
            con.close()
    
    def _check_all_metrics_present(self):
        """Verify all expected metrics are computed."""
        expected_metrics = [
            "corsi_rapm_5v5",
            "corsi_off_rapm_5v5",
            "corsi_def_rapm_5v5",
            "goals_rapm_5v5",
            "primary_assist_rapm_5v5",
            "secondary_assist_rapm_5v5",
            "xg_rapm_5v5",
            "xg_off_rapm_5v5",
            "xg_def_rapm_5v5",
            "penalties_taken_rapm_5v5",
            "penalties_drawn_rapm_5v5",
        ]
        
        con = duckdb.connect(self.db_path, read_only=True)
        
        try:
            result = con.execute("""
                SELECT DISTINCT metric_name
                FROM apm_results
                WHERE season = (SELECT MAX(season) FROM apm_results)
            """).fetchall()
            
            present = {r[0] for r in result}
            missing = [m for m in expected_metrics if m not in present]
            
            if missing:
                self.log("metrics_present", False, f"Missing metrics: {missing}", "ERROR")
            else:
                self.log("metrics_present", True, f"All {len(expected_metrics)} core metrics present")
        finally:
            con.close()
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for r in self.results if r["passed"])
        failed = len(self.results) - passed
        
        print(f"\nTotal checks: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed checks:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['check']} ({r['severity']}): {r['message']}")
        
        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="RAPM Pipeline Validator")
    parser.add_argument("--pre", action="store_true", help="Run pre-pipeline checks")
    parser.add_argument("--post", action="store_true", help="Run post-pipeline checks")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    args = parser.parse_args()
    
    if not any([args.pre, args.post, args.all]):
        args.all = True  # Default to all
    
    validator = RAPMValidator()
    all_passed = True
    
    if args.pre or args.all:
        if not validator.run_pre_checks():
            all_passed = False
    
    if args.post or args.all:
        if not validator.run_post_checks():
            all_passed = False
    
    success = validator.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

"""
Comprehensive RAPM Quality Analysis Suite

TDD-style test suite that analyzes every RAPM metric and flags:
- Outliers (players with extreme values)
- Statistical anomalies (distributions, means, std)
- Sample size issues (low games/events counts)
- Elite player sanity checks
- Cross-metric correlations

Run: python comprehensive_rapm_qa.py
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    severity: str  # "INFO", "WARNING", "ERROR", "CRITICAL"
    message: str
    details: Optional[Dict[str, Any]] = None


class RAPMQualityAnalyzer:
    """Comprehensive QA analyzer for RAPM metrics."""
    
    # Known elite players for sanity checks
    ELITE_PLAYERS = {
        # Forwards - elite scorers/playmakers
        8478402: "Connor McDavid",
        8477934: "Leon Draisaitl",
        8477492: "Nathan MacKinnon",
        8476453: "Nikita Kucherov",
        8478483: "Auston Matthews",
        8479318: "David Pastrnak",
        8477493: "Mikko Rantanen",
        8480012: "Jack Hughes",
        8479339: "Kirill Kaprizov",
        8481559: "Tim Stutzle",
        
        # Defensemen - elite
        8478420: "Cale Makar",
        8477968: "Adam Fox",
        8480069: "Quinn Hughes",
        8479323: "Miro Heiskanen",
        8477508: "Erik Karlsson",
        
        # Goalies (for reference, not in skater RAPM)
        8478009: "Andrei Vasilevskiy",
        8480382: "Igor Shesterkin",
    }
    
    # Expected ranges for different metric types
    METRIC_EXPECTED_RANGES = {
        "corsi_rapm_5v5": {"mean": (-0.1, 0.1), "std": (0.1, 0.5), "range": (-2, 2)},
        "xg_rapm_5v5": {"mean": (-0.05, 0.05), "std": (0.02, 0.15), "range": (-0.5, 0.5)},
        "goals_rapm_5v5": {"mean": (-0.05, 0.05), "std": (0.01, 0.1), "range": (-0.3, 0.3)},
        "primary_assist_rapm_5v5": {"mean": (-0.02, 0.02), "std": (0.01, 0.08), "range": (-0.25, 0.25)},
    }
    
    def __init__(self, db_path: str = "nhl_canonical.duckdb"):
        self.db_path = db_path
        self.results: List[TestResult] = []
        self.output_file = Path("comprehensive_qa_report.txt")
        
    def log(self, name: str, passed: bool, message: str, 
            severity: str = "INFO", details: Optional[Dict] = None):
        """Log a test result."""
        result = TestResult(name, passed, severity, message, details)
        self.results.append(result)
        
        status = "PASS" if passed else f"FAIL"
        icon = "✓" if passed else "✗"
        print(f"  [{icon}] [{severity:8}] {name}: {message}")
        
    def run_all_tests(self):
        """Run all comprehensive tests."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE RAPM QUALITY ANALYSIS")
        print("=" * 80)
        
        con = duckdb.connect(self.db_path, read_only=True)
        
        try:
            # Get all metrics and seasons
            metrics = self._get_all_metrics(con)
            seasons = self._get_all_seasons(con)
            
            print(f"\nAnalyzing {len(metrics)} metrics across {len(seasons)} seasons...")
            
            # Test 1: Games count consistency
            print("\n" + "-" * 60)
            print("1. GAMES COUNT CONSISTENCY")
            print("-" * 60)
            self._test_games_consistency(con, seasons)
            
            # Test 2: Events count analysis
            print("\n" + "-" * 60)
            print("2. EVENTS COUNT ANALYSIS")
            print("-" * 60)
            self._test_events_counts(con, metrics, seasons)
            
            # Test 3: Statistical distributions per metric
            print("\n" + "-" * 60)
            print("3. STATISTICAL DISTRIBUTIONS")
            print("-" * 60)
            self._test_distributions(con, metrics, seasons)
            
            # Test 4: Outlier detection
            print("\n" + "-" * 60)
            print("4. OUTLIER DETECTION")
            print("-" * 60)
            self._test_outliers(con, metrics, seasons)
            
            # Test 5: Elite player rankings
            print("\n" + "-" * 60)
            print("5. ELITE PLAYER SANITY CHECKS")
            print("-" * 60)
            self._test_elite_players(con, metrics, seasons)
            
            # Test 6: Cross-metric correlations
            print("\n" + "-" * 60)
            print("6. CROSS-METRIC CORRELATIONS")
            print("-" * 60)
            self._test_correlations(con, seasons)
            
            # Test 7: Value range checks
            print("\n" + "-" * 60)
            print("7. VALUE RANGE CHECKS")
            print("-" * 60)
            self._test_value_ranges(con, metrics, seasons)
            
            # Test 8: TOI sanity checks
            print("\n" + "-" * 60)
            print("8. TOI SANITY CHECKS")
            print("-" * 60)
            self._test_toi_sanity(con, seasons)
            
        finally:
            con.close()
        
        # Print summary
        self._print_summary()
        self._save_report()
        
    def _get_all_metrics(self, con) -> List[str]:
        """Get all unique metric names."""
        result = con.execute(
            "SELECT DISTINCT metric_name FROM apm_results ORDER BY metric_name"
        ).fetchall()
        return [r[0] for r in result]
    
    def _get_all_seasons(self, con) -> List[str]:
        """Get all unique seasons."""
        result = con.execute(
            "SELECT DISTINCT season FROM apm_results ORDER BY season DESC"
        ).fetchall()
        return [r[0] for r in result]
    
    def _test_games_consistency(self, con, seasons: List[str]):
        """Test that all metrics use same games_count per season."""
        for season in seasons:
            result = con.execute(f"""
                SELECT metric_name, MAX(games_count) as games
                FROM apm_results
                WHERE season = '{season}'
                GROUP BY metric_name
                ORDER BY games DESC
            """).fetchall()
            
            if not result:
                continue
                
            games_values = set(g for m, g in result)
            max_games = max(games_values)
            min_games = min(games_values)
            
            if len(games_values) > 1:
                low_games_metrics = [m for m, g in result if g < max_games * 0.5]
                self.log(
                    f"games_consistency_{season}",
                    False,
                    f"Inconsistent: {min_games}-{max_games} games. {len(low_games_metrics)} metrics have <50% of max games",
                    "ERROR",
                    {"low_games_metrics": low_games_metrics[:5]}
                )
            else:
                self.log(
                    f"games_consistency_{season}",
                    True,
                    f"Consistent: {max_games} games for all metrics",
                    "INFO"
                )
    
    def _test_events_counts(self, con, metrics: List[str], seasons: List[str]):
        """Test events counts are reasonable for each metric type."""
        for season in seasons[-2:]:  # Check last 2 seasons
            for metric in metrics[:10]:  # Sample of metrics
                result = con.execute(f"""
                    SELECT MAX(events_count) as events, MAX(games_count) as games
                    FROM apm_results
                    WHERE season = '{season}' AND metric_name = '{metric}'
                """).fetchone()
                
                if not result or result[0] is None:
                    continue
                    
                events, games = result
                
                # Expected events per game varies by metric type
                if "corsi" in metric and "5v5" in metric and "off" not in metric and "def" not in metric:
                    expected_per_game = 100  # ~100 Corsi events per game
                elif "goal" in metric:
                    expected_per_game = 4  # ~4 5v5 goals per game
                elif "assist" in metric:
                    expected_per_game = 3  # ~3 assists per game
                elif "xg" in metric:
                    expected_per_game = 40  # ~40 shots with xG per game
                else:
                    continue
                
                expected_total = games * expected_per_game * 0.3  # Allow 70% lower than expected
                
                if events < expected_total:
                    self.log(
                        f"events_{metric}_{season}",
                        False,
                        f"Low events: {events} (expected >={int(expected_total)} for {games} games)",
                        "WARNING",
                        {"events": events, "games": games}
                    )
    
    def _test_distributions(self, con, metrics: List[str], seasons: List[str]):
        """Test statistical distributions are reasonable."""
        latest_season = seasons[0] if seasons else None
        if not latest_season:
            return
            
        for metric in metrics:
            result = con.execute(f"""
                SELECT 
                    AVG(value) as mean_val,
                    STDDEV(value) as std_val,
                    MIN(value) as min_val,
                    MAX(value) as max_val,
                    COUNT(*) as n
                FROM apm_results
                WHERE season = '{latest_season}' AND metric_name = '{metric}'
            """).fetchone()
            
            if not result or result[0] is None:
                continue
                
            mean_val, std_val, min_val, max_val, n = result
            
            # Check mean is near zero (RAPM should be balanced)
            if abs(mean_val) > 0.15:
                self.log(
                    f"dist_mean_{metric}",
                    False,
                    f"Mean far from zero: {mean_val:.4f}",
                    "WARNING"
                )
            
            # Check std is not too narrow (would indicate no variation)
            if std_val is not None and std_val < 0.001:
                self.log(
                    f"dist_std_{metric}",
                    False,
                    f"Std too narrow: {std_val:.6f} (no player differentiation)",
                    "ERROR"
                )
            
            # Check range is reasonable
            value_range = max_val - min_val if max_val and min_val else 0
            if value_range < 0.01:
                self.log(
                    f"dist_range_{metric}",
                    False,
                    f"Range too narrow: {value_range:.6f}",
                    "ERROR"
                )
    
    def _test_outliers(self, con, metrics: List[str], seasons: List[str]):
        """Detect statistical outliers."""
        latest_season = seasons[0] if seasons else None
        if not latest_season:
            return
            
        outlier_count = 0
        for metric in metrics[:15]:  # Top metrics
            result = con.execute(f"""
                WITH stats AS (
                    SELECT 
                        AVG(value) as mean_val,
                        STDDEV(value) as std_val
                    FROM apm_results
                    WHERE season = '{latest_season}' AND metric_name = '{metric}'
                )
                SELECT a.player_id, a.value, 
                       (a.value - s.mean_val) / NULLIF(s.std_val, 0) as z_score
                FROM apm_results a, stats s
                WHERE a.season = '{latest_season}' 
                  AND a.metric_name = '{metric}'
                  AND ABS((a.value - s.mean_val) / NULLIF(s.std_val, 0)) > 4
                ORDER BY ABS((a.value - s.mean_val) / NULLIF(s.std_val, 0)) DESC
                LIMIT 5
            """).fetchall()
            
            if result:
                outlier_count += len(result)
                for pid, val, z in result:
                    player_name = self.ELITE_PLAYERS.get(pid, f"Player #{pid}")
                    self.log(
                        f"outlier_{metric}",
                        False,
                        f"Extreme value: {player_name} = {val:.4f} (z={z:.1f})",
                        "INFO" if abs(z) < 5 else "WARNING"
                    )
        
        if outlier_count == 0:
            self.log("outliers_check", True, "No extreme outliers (z>4) detected", "INFO")
    
    def _test_elite_players(self, con, metrics: List[str], seasons: List[str]):
        """Check elite players rank reasonably."""
        latest_season = seasons[0] if seasons else None
        if not latest_season:
            return
        
        key_metrics = [
            ("corsi_rapm_5v5", "Corsi", 100),  # Should be in top 100
            ("xg_rapm_5v5", "xG", 100),
            ("xg_off_rapm_5v5", "Offensive xG", 50),  # Top forwards in top 50
            ("primary_assist_rapm_5v5", "Primary Assists", 150),
        ]
        
        elite_forward_ids = [8478402, 8477934, 8477492, 8476453]  # McDavid, Draisaitl, MacKinnon, Kucherov
        
        for metric, label, expected_rank in key_metrics:
            result = con.execute(f"""
                WITH ranked AS (
                    SELECT player_id, value,
                           ROW_NUMBER() OVER (ORDER BY value DESC) as rank
                    FROM apm_results
                    WHERE season = '{latest_season}' AND metric_name = '{metric}'
                )
                SELECT player_id, rank FROM ranked
                WHERE player_id IN ({','.join(map(str, elite_forward_ids))})
            """).fetchall()
            
            if not result:
                self.log(
                    f"elite_{metric}",
                    False,
                    f"No elite forwards found in {label} rankings",
                    "WARNING"
                )
                continue
            
            rankings = {pid: rank for pid, rank in result}
            in_expected = sum(1 for r in rankings.values() if r <= expected_rank)
            
            details = ", ".join([
                f"{self.ELITE_PLAYERS.get(pid, pid)}:#{r}" 
                for pid, r in sorted(rankings.items(), key=lambda x: x[1])
            ])
            
            if in_expected >= 2:  # At least 2 of 4 elite forwards in expected range
                self.log(
                    f"elite_{metric}",
                    True,
                    f"{in_expected}/4 elite forwards in top {expected_rank}: {details}",
                    "INFO"
                )
            else:
                self.log(
                    f"elite_{metric}",
                    False,
                    f"Only {in_expected}/4 elite forwards in top {expected_rank}: {details}",
                    "WARNING"
                )
    
    def _test_correlations(self, con, seasons: List[str]):
        """Test expected correlations between metrics."""
        latest_season = seasons[0] if seasons else None
        if not latest_season:
            return
        
        # Expected positive correlations
        correlations_to_check = [
            ("corsi_rapm_5v5", "xg_rapm_5v5", 0.3),  # Should correlate
            ("corsi_off_rapm_5v5", "xg_off_rapm_5v5", 0.4),  # Offensive metrics correlate
            ("corsi_def_rapm_5v5", "xg_def_rapm_5v5", 0.4),  # Defensive metrics correlate
        ]
        
        for metric1, metric2, min_corr in correlations_to_check:
            result = con.execute(f"""
                SELECT CORR(a.value, b.value) as correlation
                FROM apm_results a
                JOIN apm_results b ON a.player_id = b.player_id AND a.season = b.season
                WHERE a.season = '{latest_season}'
                  AND a.metric_name = '{metric1}'
                  AND b.metric_name = '{metric2}'
            """).fetchone()
            
            if not result or result[0] is None:
                self.log(
                    f"corr_{metric1}_vs_{metric2}",
                    False,
                    f"Could not compute correlation",
                    "WARNING"
                )
                continue
            
            corr = result[0]
            passed = corr >= min_corr
            self.log(
                f"corr_{metric1}_vs_{metric2}",
                passed,
                f"Correlation: {corr:.3f} (expected >= {min_corr})",
                "INFO" if passed else "WARNING"
            )
    
    def _test_value_ranges(self, con, metrics: List[str], seasons: List[str]):
        """Check values are in expected ranges."""
        latest_season = seasons[0] if seasons else None
        if not latest_season:
            return
        
        for metric, ranges in self.METRIC_EXPECTED_RANGES.items():
            result = con.execute(f"""
                SELECT MIN(value), MAX(value), AVG(value), STDDEV(value)
                FROM apm_results
                WHERE season = '{latest_season}' AND metric_name = '{metric}'
            """).fetchone()
            
            if not result or result[0] is None:
                continue
            
            min_val, max_val, mean_val, std_val = result
            
            # Check mean
            if not (ranges["mean"][0] <= mean_val <= ranges["mean"][1]):
                self.log(
                    f"range_{metric}_mean",
                    False,
                    f"Mean {mean_val:.4f} outside expected {ranges['mean']}",
                    "WARNING"
                )
            
            # Check range
            if min_val < ranges["range"][0] * 2 or max_val > ranges["range"][1] * 2:
                self.log(
                    f"range_{metric}_bounds",
                    False,
                    f"Values [{min_val:.3f}, {max_val:.3f}] exceed 2x expected range",
                    "WARNING"
                )
    
    def _test_toi_sanity(self, con, seasons: List[str]):
        """Check TOI values are reasonable."""
        latest_season = seasons[0] if seasons else None
        if not latest_season:
            return
        
        # Check if toi_seconds is in the table
        result = con.execute(f"""
            SELECT MIN(toi_seconds), MAX(toi_seconds), AVG(toi_seconds)
            FROM apm_results
            WHERE season = '{latest_season}' AND metric_name = 'corsi_rapm_5v5'
        """).fetchone()
        
        if not result or result[0] is None:
            self.log("toi_check", False, "No TOI data available", "WARNING")
            return
        
        min_toi, max_toi, avg_toi = result
        
        # Convert to minutes
        min_toi_min = min_toi / 60 if min_toi else 0
        max_toi_min = max_toi / 60 if max_toi else 0
        avg_toi_min = avg_toi / 60 if avg_toi else 0
        
        # Expected: min ~60 min (to be included), max ~2000 min (top-line forward full season)
        if min_toi_min < 10:
            self.log(
                "toi_min",
                False,
                f"Some players have very low TOI: {min_toi_min:.0f} min",
                "INFO"
            )
        
        if max_toi_min > 2500:
            self.log(
                "toi_max",
                False,
                f"Max TOI seems too high: {max_toi_min:.0f} min (>41 hrs)",
                "WARNING"
            )
        elif max_toi_min < 500:
            self.log(
                "toi_max",
                False,
                f"Max TOI seems too low: {max_toi_min:.0f} min (incomplete season?)",
                "WARNING"
            )
        else:
            self.log(
                "toi_range",
                True,
                f"TOI range reasonable: {min_toi_min:.0f} - {max_toi_min:.0f} min (avg {avg_toi_min:.0f})",
                "INFO"
            )
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        by_severity = defaultdict(int)
        for r in self.results:
            if not r.passed:
                by_severity[r.severity] += 1
        
        print(f"\nTotal tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if by_severity:
            print("\nFailures by severity:")
            for sev in ["CRITICAL", "ERROR", "WARNING", "INFO"]:
                if sev in by_severity:
                    print(f"  {sev}: {by_severity[sev]}")
        
        # List critical/error failures
        critical_errors = [r for r in self.results if not r.passed and r.severity in ["CRITICAL", "ERROR"]]
        if critical_errors:
            print("\nCritical/Error issues:")
            for r in critical_errors[:10]:
                print(f"  - {r.name}: {r.message}")
    
    def _save_report(self):
        """Save detailed report to file."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("COMPREHENSIVE RAPM QUALITY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            total = len(self.results)
            passed = sum(1 for r in self.results if r.passed)
            f.write(f"Total tests: {total}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Failed: {total - passed}\n\n")
            
            # All results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 80 + "\n\n")
            
            for r in self.results:
                status = "PASS" if r.passed else "FAIL"
                f.write(f"[{status}] [{r.severity:8}] {r.name}\n")
                f.write(f"    {r.message}\n")
                if r.details:
                    f.write(f"    Details: {json.dumps(r.details)}\n")
                f.write("\n")
        
        print(f"\nDetailed report saved to: {self.output_file}")


def main():
    analyzer = RAPMQualityAnalyzer()
    analyzer.run_all_tests()


if __name__ == "__main__":
    main()

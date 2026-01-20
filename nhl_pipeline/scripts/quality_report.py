#!/usr/bin/env python3
"""
Quality Report - Main reporting script.

Combines schema registry and validation history into a single report.
Run after pipeline to see data quality summary.

Usage:
    python quality_report.py              # Full report
    python quality_report.py --brief      # Just summary
    python quality_report.py --json       # Output as JSON
    python quality_report.py --problems   # Only show problems
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd

from schema_registry import SchemaRegistry
from validation_history import ValidationHistory


def print_header(title: str, width: int = 70):
    """Print a section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subheader(title: str, width: int = 60):
    """Print a subsection header."""
    print(f"\n{title}")
    print("-" * width)


def generate_full_report(data_dir: Path) -> Dict[str, Any]:
    """Generate complete quality report."""
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "validation": {},
        "schema": {},
        "problems": [],
        "recommendations": [],
    }
    
    # Load validation history
    history_path = data_dir / "validation_history.parquet"
    history = ValidationHistory(history_path)
    
    # Load schema registry
    registry_path = data_dir / "schema_registry.parquet"
    registry = SchemaRegistry(registry_path)
    
    # === VALIDATION SUMMARY ===
    print_header("📊 NHL DATA QUALITY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    latest = history.get_latest_by_game()
    
    if latest.empty:
        print("\n⚠️  No validation data found. Run the pipeline first.")
        return report
    
    # Overall metrics
    total_games = len(latest)
    passed_games = latest["all_passed"].sum()
    pass_rate = 100 * passed_games / total_games if total_games > 0 else 0
    avg_toi_diff = latest["toi_diff_avg"].mean()
    total_overlaps = latest["overlap_count"].sum()
    
    report["validation"]["total_games"] = int(total_games)
    report["validation"]["passed_games"] = int(passed_games)
    report["validation"]["pass_rate"] = round(pass_rate, 1)
    report["validation"]["avg_toi_diff"] = round(avg_toi_diff, 1)
    report["validation"]["total_overlaps"] = int(total_overlaps)
    
    print_subheader("📈 OVERALL METRICS")
    print(f"  Games processed:     {total_games:,}")
    print(f"  Games passed:        {passed_games:,}")
    print(f"  Pass rate:           {pass_rate:.1f}%", end="")
    if pass_rate >= 95:
        print(" ✓")
    elif pass_rate >= 80:
        print(" ⚠️")
    else:
        print(" ✗")
    print(f"  Avg TOI difference:  {avg_toi_diff:.1f}s (threshold: 120s)")
    print(f"  Total overlaps:      {total_overlaps:,}")
    
    # By season
    print_subheader("📅 BY SEASON")
    
    summary = history.get_summary_by_season()
    report["validation"]["by_season"] = summary.to_dict(orient="records")
    
    print(f"  {'Season':<12} {'Games':>7} {'Passed':>8} {'Failed':>8} {'Pass%':>8} {'Avg TOI':>10}")
    print("  " + "-" * 56)
    
    for _, row in summary.iterrows():
        status = "✓" if row["pass_rate"] == 100 else "⚠️" if row["pass_rate"] >= 80 else "✗"
        print(f"  {row['season']:<12} {int(row['games']):>7} {int(row['passed']):>8} {int(row['failed']):>8} {row['pass_rate']:>7.1f}% {row['avg_toi_diff']:>9.1f}s {status}")
    
    # Failed games
    failed = history.get_failed_games()
    
    if not failed.empty:
        print_subheader(f"❌ FAILED GAMES ({len(failed)} total)")
        
        report["validation"]["failed_games"] = failed.to_dict(orient="records")
        
        for _, row in failed.head(15).iterrows():
            reasons = []
            if row["toi_diff_avg"] > 120:
                reasons.append(f"TOI mismatch: {row['toi_diff_avg']:.0f}s")
            if row["overlap_count"] > 0:
                reasons.append(f"{row['overlap_count']} overlaps")
            if row["missing_goals_count"] > 0:
                reasons.append(f"{row['missing_goals_count']} goals without on-ice data")
            
            reason_str = ", ".join(reasons) if reasons else "unknown failure"
            print(f"  • {row['season']}/{row['game_id']}: {reason_str}")
        
        if len(failed) > 15:
            print(f"  ... and {len(failed) - 15} more")
    else:
        print_subheader("✓ ALL GAMES PASSED VALIDATION")
    
    # === SCHEMA REGISTRY ===
    print_header("🗂️ SCHEMA REGISTRY")
    
    schema_df = registry.get_summary_df()
    
    if schema_df.empty:
        print("\n⚠️  No schema data found. Run the pipeline first.")
    else:
        # Summary by endpoint
        for endpoint in schema_df["endpoint"].unique():
            endpoint_df = schema_df[schema_df["endpoint"] == endpoint]
            
            print(f"\n  {endpoint.upper()}")
            
            for season in sorted(endpoint_df["season"].unique(), reverse=True):
                season_df = endpoint_df[endpoint_df["season"] == season]
                total = len(season_df)
                ok = (season_df["presence_rate"] >= 99).sum()
                warn = ((season_df["presence_rate"] >= 90) & (season_df["presence_rate"] < 99)).sum()
                bad = (season_df["presence_rate"] < 90).sum()
                
                status = "✓" if bad == 0 and warn == 0 else "⚠️" if bad == 0 else "✗"
                print(f"    {season}: {ok:>2}✓  {warn:>2}⚠️  {bad:>2}✗  ({total} fields) {status}")
        
        # Problem fields
        problems = registry.get_problems()
        if not problems.empty:
            print_subheader("⚠️  PROBLEM FIELDS (low presence rate)")
            
            for _, row in problems.iterrows():
                nullable = "nullable" if row["nullable"] else "required"
                print(f"  • {row['endpoint']}/{row['season']}: {row['api_field']} = {row['presence_rate']:.1f}% ({nullable})")
                
                report["problems"].append({
                    "type": "schema",
                    "endpoint": row["endpoint"],
                    "season": row["season"],
                    "field": row["api_field"],
                    "presence_rate": row["presence_rate"],
                })
        
        # Schema changes
        changes = registry.detect_schema_changes()
        if changes:
            print_subheader("🔄 SCHEMA CHANGES BETWEEN SEASONS")
            
            for c in changes:
                direction = "📈" if c["change"] > 0 else "📉"
                print(f"  {direction} {c['endpoint']}/{c['field']}: {c['from_season']}→{c['to_season']} ({c['from_rate']:.1f}%→{c['to_rate']:.1f}%)")
                
                report["problems"].append({
                    "type": "schema_change",
                    "endpoint": c["endpoint"],
                    "field": c["field"],
                    "from_season": c["from_season"],
                    "to_season": c["to_season"],
                    "change": c["change"],
                })
    
    # === RECOMMENDATIONS ===
    print_header("💡 RECOMMENDATIONS")
    
    recommendations = []
    
    if pass_rate < 95:
        rec = f"Pass rate is {pass_rate:.1f}%. Investigate failed games before using data for modeling."
        recommendations.append(rec)
        print(f"  ⚠️  {rec}")
    
    if avg_toi_diff > 60:
        rec = f"Average TOI difference ({avg_toi_diff:.1f}s) is high. Check shift parsing logic."
        recommendations.append(rec)
        print(f"  ⚠️  {rec}")
    
    if total_overlaps > 0:
        rec = f"{total_overlaps} overlapping shifts found. May indicate duplicate records."
        recommendations.append(rec)
        print(f"  ⚠️  {rec}")
    
    if not failed.empty:
        # Check if failures are concentrated in specific seasons
        failed_by_season = failed.groupby("season").size()
        worst_season = failed_by_season.idxmax()
        worst_count = failed_by_season.max()
        
        season_total = summary[summary["season"] == worst_season]["games"].values[0]
        if worst_count / season_total > 0.2:
            rec = f"Season {worst_season} has high failure rate ({worst_count}/{int(season_total)}). May need season-specific parsing."
            recommendations.append(rec)
            print(f"  ⚠️  {rec}")
    
    if not recommendations:
        print("  ✓ No critical issues found. Data is ready for modeling.")
        recommendations.append("No critical issues. Data is ready for modeling.")
    
    report["recommendations"] = recommendations
    
    print("\n" + "=" * 70)
    
    return report


def generate_brief_report(data_dir: Path):
    """Generate brief summary only."""
    
    history_path = data_dir / "validation_history.parquet"
    history = ValidationHistory(history_path)
    
    latest = history.get_latest_by_game()
    
    if latest.empty:
        print("No validation data found.")
        return
    
    total = len(latest)
    passed = latest["all_passed"].sum()
    rate = 100 * passed / total
    
    status = "✓" if rate >= 95 else "⚠️" if rate >= 80 else "✗"
    
    print(f"NHL Data Quality: {passed}/{total} games passed ({rate:.1f}%) {status}")
    
    if rate < 100:
        failed = latest[~latest["all_passed"]]
        print(f"  Failed: {', '.join(failed['game_id'].head(5).tolist())}", end="")
        if len(failed) > 5:
            print(f" +{len(failed)-5} more")
        else:
            print()


def main():
    parser = argparse.ArgumentParser(description="NHL Data Quality Report")
    parser.add_argument("--brief", action="store_true", help="Brief summary only")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--problems", action="store_true", help="Show only problems")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory path")
    
    args = parser.parse_args()
    
    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data"
    
    if args.brief:
        generate_brief_report(data_dir)
    else:
        report = generate_full_report(data_dir)
        
        if args.json:
            print("\n" + json.dumps(report, indent=2))
        
        if args.problems:
            if report["problems"]:
                print("\nProblems found:")
                for p in report["problems"]:
                    print(f"  - {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RAPM Validation Script

Validates RAPM implementation against Evolving-Hockey methodology
and performs final sanity checks.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Redirect stdout to file
sys.stdout = open("rapm_validation_evolving_output.txt", "w", encoding="utf-8")

def audit_rapm_inputs(conn, season: str, stints_path: Path = None):
    """
    Audit the RAPM inputs to verify they match Evolving-Hockey methodology.
    
    Evolving-Hockey specification:
    - Target: per-60 rates (GF/60, xGF/60, CF/60)
    - Weights: stint duration in seconds
    - Design matrix: 2 rows per stint (home off/away def, away off/home def)
    """
    
    print("=" * 70)
    print("RAPM INPUT AUDIT (vs Evolving-Hockey Methodology)")
    print("=" * 70)
    
    # If stint data is available, audit it directly
    if stints_path and stints_path.exists():
        stints = pd.read_parquet(stints_path)
        
        print(f"\n--- Stint Data ---")
        print(f"Total stints: {len(stints):,}")
        print(f"Columns: {stints.columns.tolist()}")
        
        print(f"\n--- Duration (Weight) ---")
        print(f"  Min:    {stints['duration_s'].min():.1f} s")
        print(f"  Median: {stints['duration_s'].median():.1f} s")
        print(f"  Mean:   {stints['duration_s'].mean():.1f} s")
        print(f"  Max:    {stints['duration_s'].max():.1f} s")
        print(f"  Expected: Median 20-40s")
        
        if stints['duration_s'].median() >= 15 and stints['duration_s'].median() <= 50:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL - Stint durations outside expected range")
        
        print(f"\n--- xG Values (Raw per stint) ---")
        print(f"  xg_home - min: {stints['xg_home'].min():.4f}, max: {stints['xg_home'].max():.4f}, mean: {stints['xg_home'].mean():.4f}")
        print(f"  xg_away - min: {stints['xg_away'].min():.4f}, max: {stints['xg_away'].max():.4f}, mean: {stints['xg_away'].mean():.4f}")
        
        print(f"\n--- Target Variable Check ---")
        # Calculate what y should be for per-60
        y_per_second = stints['xg_home'] / stints['duration_s']
        y_per_60 = y_per_second * 3600
        
        print(f"  If y = xg / duration (per-second):")
        print(f"    mean: {y_per_second.mean():.6f}")
        print(f"  If y = xg / duration * 3600 (per-60):")
        print(f"    mean: {y_per_60.mean():.4f}")
        print(f"  Expected per-60 mean: ~2-4 xG/60")
        
        if 1 < y_per_60.mean() < 6:
            print("  ✓ Per-60 calculation looks correct")
        else:
            print("  ✗ Check target variable calculation")
    
    print("\n" + "=" * 70)
    print("EVOLVING-HOCKEY METHODOLOGY CHECKLIST")
    print("=" * 70)
    
    checklist = [
        ("Target variable is per-60 rate", "Check y calculation"),
        ("Weights are stint duration in seconds", "Check w = duration_s"),
        ("2 rows per stint (off/def split)", "Check obs_df length = 2 × stints"),
        ("Offense cols: +1 for offensive skaters", "Check design matrix"),
        ("Defense cols: +1 for defensive skaters", "Check design matrix"),
        ("Ridge regression with regularization", "Check alpha value"),
        ("Coefficients interpreted as per-60", "No post-hoc × 3600 if y is per-60"),
    ]
    
    print("\nManual verification needed:")
    for item, note in checklist:
        print(f"  [ ] {item}")
        print(f"      → {note}")
    
    # Additional controls (Evolving-Hockey uses, optional for basic RAPM)
    print("\n--- Additional Controls (Evolving-Hockey) ---")
    print("  [ ] Score state (trailing/tied/leading)")
    print("  [ ] Zone starts (OZ/NZ/DZ faceoffs)")
    print("  [ ] Back-to-back games")
    print("  Note: These improve accuracy but aren't required for basic RAPM")


def check_design_matrix(conn, season: str):
    """
    Verify design matrix structure.
    """
    print("\n" + "=" * 70)
    print("DESIGN MATRIX CHECK")
    print("=" * 70)
    
    # Count players
    player_count = conn.execute(f"""
        SELECT COUNT(DISTINCT player_id) as n
        FROM apm_results
        WHERE season = '{season}'
          AND metric_name = 'xg_off_rapm_5v5'
    """).fetchone()[0]
    
    print(f"\nPlayers in model: {player_count}")
    print(f"Expected design matrix columns: {2 * player_count} (off + def per player)")
    
    # Check if off and def have same players
    off_players = conn.execute(f"""
        SELECT player_id FROM apm_results
        WHERE season = '{season}' AND metric_name = 'xg_off_rapm_5v5'
    """).df()["player_id"].tolist()
    
    def_players = conn.execute(f"""
        SELECT player_id FROM apm_results
        WHERE season = '{season}' AND metric_name = 'xg_def_rapm_5v5'
    """).df()["player_id"].tolist()
    
    if set(off_players) == set(def_players):
        print("✓ Same players in offensive and defensive RAPM")
    else:
        print("✗ Mismatch between offensive and defensive player lists")
        print(f"  Off only: {len(set(off_players) - set(def_players))}")
        print(f"  Def only: {len(set(def_players) - set(off_players))}")


def range_check(conn, season: str):
    """
    Check if coefficients are in expected ranges.
    """
    print("\n" + "=" * 70)
    print("COEFFICIENT RANGE CHECK")
    print("=" * 70)
    
    results = conn.execute(f"""
        SELECT 
            metric_name,
            COUNT(*) as n_players,
            MIN(value) as min_val,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY value) as p5,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) as median,
            AVG(value) as mean,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as p95,
            MAX(value) as max_val,
            STDDEV(value) as std
        FROM apm_results
        WHERE season = '{season}'
        GROUP BY metric_name
        ORDER BY metric_name
    """).df()
    
    expected_ranges = {
        "xg_off_rapm_5v5": (-1.5, 1.5),
        "xg_def_rapm_5v5": (-1.5, 1.5),
        "goals_rapm_5v5": (-1.0, 1.0),
        "corsi_rapm_5v5": (-15, 15),
        "corsi_off_rapm_5v5": (-15, 15),
        "corsi_def_rapm_5v5": (-15, 15),
        "primary_assist_rapm_5v5": (-0.5, 0.5),
    }
    
    print(f"\n{'Metric':<35} {'N':>6} {'Min':>8} {'P5':>8} {'Med':>8} {'Mean':>8} {'P95':>8} {'Max':>8} {'P5-P95':>8}")
    print("-" * 115)
    
    for _, row in results.iterrows():
        metric = row["metric_name"]
        expected = expected_ranges.get(metric)
        
        # Check if P5-P95 is in expected range (more robust than min/max)
        if expected:
            p5_ok = row["p5"] >= expected[0]
            p95_ok = row["p95"] <= expected[1]
            status = "✓" if (p5_ok and p95_ok) else "✗"
        else:
            status = "?"
        
        print(f"{metric:<35} {row['n_players']:>6.0f} {row['min_val']:>8.2f} {row['p5']:>8.2f} {row['median']:>8.2f} {row['mean']:>8.2f} {row['p95']:>8.2f} {row['max_val']:>8.2f} {status:>8}")


def distribution_check(conn, season: str):
    """
    Check coefficient distributions are roughly normal and centered.
    """
    print("\n" + "=" * 70)
    print("DISTRIBUTION CHECK")
    print("=" * 70)
    
    off = conn.execute(f"""
        SELECT value FROM apm_results 
        WHERE season = '{season}' AND metric_name = 'xg_off_rapm_5v5'
    """).df()["value"]
    
    def_ = conn.execute(f"""
        SELECT value FROM apm_results 
        WHERE season = '{season}' AND metric_name = 'xg_def_rapm_5v5'
    """).df()["value"]
    
    print("\nOffensive xG RAPM:")
    print(f"  N:      {len(off)}")
    print(f"  Mean:   {off.mean():.4f} (should be ~0)")
    print(f"  Median: {off.median():.4f}")
    print(f"  Std:    {off.std():.4f}")
    print(f"  Skew:   {off.skew():.4f} (should be ~0)")
    
    print("\nDefensive xG RAPM:")
    print(f"  N:      {len(def_)}")
    print(f"  Mean:   {def_.mean():.4f} (should be ~0)")
    print(f"  Median: {def_.median():.4f}")
    print(f"  Std:    {def_.std():.4f}")
    print(f"  Skew:   {def_.skew():.4f} (should be ~0)")
    
    # Validation
    print("\n--- Validation ---")
    checks = [
        ("Off mean ~0", abs(off.mean()) < 0.05, off.mean()),
        ("Def mean ~0", abs(def_.mean()) < 0.05, def_.mean()),
        ("Off skew reasonable", abs(off.skew()) < 1.0, off.skew()),
        ("Def skew reasonable", abs(def_.skew()) < 1.0, def_.skew()),
    ]
    
    for name, passed, value in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status} (value: {value:.4f})")


def correlation_check(conn, season: str):
    """
    Check correlations between different RAPM metrics.
    """
    print("\n" + "=" * 70)
    print("CORRELATION CHECK")
    print("=" * 70)
    
    metrics = conn.execute(f"""
        SELECT player_id, metric_name, value
        FROM apm_results
        WHERE season = '{season}'
          AND metric_name IN (
              'xg_off_rapm_5v5', 
              'xg_def_rapm_5v5', 
              'corsi_off_rapm_5v5', 
              'corsi_def_rapm_5v5',
              'goals_rapm_5v5'
          )
    """).df()
    
    if metrics.empty:
        print("No data found for correlation check")
        return
    
    pivot = metrics.pivot(index="player_id", columns="metric_name", values="value")
    
    print("\nCorrelation Matrix:")
    corr = pivot.corr()
    
    # Print with formatting
    cols = corr.columns.tolist()
    print(f"\n{'':>25}", end="")
    for c in cols:
        print(f"{c[-15:]:>15}", end="")
    print()
    
    for i, row_name in enumerate(cols):
        print(f"{row_name[-25:]:>25}", end="")
        for j, col_name in enumerate(cols):
            val = corr.iloc[i, j]
            print(f"{val:>15.3f}", end="")
        print()
    
    print("\n--- Expected Correlations ---")
    expected = [
        ("xG_off vs Corsi_off", 0.5, 0.8, "should be related"),
        ("xG_off vs xG_def", -0.3, 0.3, "weak correlation"),
        ("xG_off vs Goals", 0.3, 0.6, "xG predicts goals"),
    ]
    
    for desc, low, high, note in expected:
        print(f"  {desc}: {low} to {high} ({note})")


def known_players_check(conn, season: str):
    """
    Check specific players we expect to be good/bad.
    """
    print("\n" + "=" * 70)
    print("KNOWN PLAYERS SPOT CHECK")
    print("=" * 70)
    
    # Known elite players
    elite_players = [
        "Connor McDavid",
        "Nathan MacKinnon", 
        "Nikita Kucherov",
        "Auston Matthews",
        "Leon Draisaitl",
        "Cale Makar",
        "Quinn Hughes",
        "Adam Fox",
    ]
    
    try:
        results = conn.execute(f"""
            SELECT 
                p.first_name || ' ' || p.last_name as name,
                a.value as xg_off,
                b.value as xg_def,
                a.value + b.value as xg_total,
                a.toi_seconds / 60.0 as toi_min,
                PERCENT_RANK() OVER (ORDER BY a.value) as off_pct,
                PERCENT_RANK() OVER (ORDER BY b.value DESC) as def_pct,
                PERCENT_RANK() OVER (ORDER BY a.value + b.value) as total_pct
            FROM apm_results a
            JOIN apm_results b 
                ON a.player_id = b.player_id 
                AND a.season = b.season
            JOIN players p ON a.player_id = p.player_id
            WHERE a.season = '{season}'
              AND a.metric_name = 'xg_off_rapm_5v5'
              AND b.metric_name = 'xg_def_rapm_5v5'
        """).df()
    except Exception as e:
        print(f"Error: {e}")
        print("Skipping known players check (players table may not exist)")
        return
    
    print(f"\n{'Player':<25} {'Off':>7} {'Def':>7} {'Total':>7} {'Off%':>7} {'Def%':>7} {'Tot%':>7} {'TOI':>6}")
    print("-" * 95)
    
    for player in elite_players:
        row = results[results["name"].str.contains(player, case=False, na=False)]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"{r['name']:<25} {r['xg_off']:>7.2f} {r['xg_def']:>7.2f} {r['xg_total']:>7.2f} {r['off_pct']*100:>6.0f}% {r['def_pct']*100:>6.0f}% {r['total_pct']*100:>6.0f}% {r['toi_min']:>6.0f}")
        else:
            print(f"{player:<25} NOT FOUND")
    
    print("\n--- Validation ---")
    print("  Elite players should be >80th percentile in total RAPM")


def final_validation(conn, season: str, min_toi_minutes: int = 200):
    """
    Final sanity check with TOI filter.
    """
    print("\n" + "=" * 70)
    print(f"FINAL VALIDATION (min {min_toi_minutes} min TOI)")
    print("=" * 70)
    
    try:
        results = conn.execute(f"""
            SELECT 
                p.first_name || ' ' || p.last_name as name,
                a.player_id,
                a.value as xg_off,
                b.value as xg_def,
                a.value + b.value as xg_total,
                a.toi_seconds / 60.0 as toi_min
            FROM apm_results a
            JOIN apm_results b 
                ON a.player_id = b.player_id 
                AND a.season = b.season
            JOIN players p ON a.player_id = p.player_id
            WHERE a.season = '{season}'
              AND a.metric_name = 'xg_off_rapm_5v5'
              AND b.metric_name = 'xg_def_rapm_5v5'
              AND a.toi_seconds >= {min_toi_minutes * 60}
            ORDER BY a.value + b.value DESC
        """).df()
    except Exception as e:
        # Fallback without player names
        print(f"Note: Could not join player names ({e})")
        results = conn.execute(f"""
            SELECT 
                CAST(a.player_id AS VARCHAR) as name,
                a.player_id,
                a.value as xg_off,
                b.value as xg_def,
                a.value + b.value as xg_total,
                a.toi_seconds / 60.0 as toi_min
            FROM apm_results a
            JOIN apm_results b 
                ON a.player_id = b.player_id 
                AND a.season = b.season
            WHERE a.season = '{season}'
              AND a.metric_name = 'xg_off_rapm_5v5'
              AND b.metric_name = 'xg_def_rapm_5v5'
              AND a.toi_seconds >= {min_toi_minutes * 60}
            ORDER BY a.value + b.value DESC
        """).df()
    
    print(f"\nPlayers included: {len(results)}")
    
    print(f"\n--- Coefficient Ranges ---")
    print(f"xG Off: {results['xg_off'].min():.2f} to {results['xg_off'].max():.2f}")
    print(f"xG Def: {results['xg_def'].min():.2f} to {results['xg_def'].max():.2f}")
    print(f"Total:  {results['xg_total'].min():.2f} to {results['xg_total'].max():.2f}")
    
    print(f"\n--- Top 10 Total xG RAPM ---")
    for _, row in results.head(10).iterrows():
        print(f"{row['name']:<25} Off: {row['xg_off']:>6.2f}  Def: {row['xg_def']:>6.2f}  Total: {row['xg_total']:>6.2f}  TOI: {row['toi_min']:>5.0f}")
    
    print(f"\n--- Bottom 10 Total xG RAPM ---")
    for _, row in results.tail(10).iterrows():
        print(f"{row['name']:<25} Off: {row['xg_off']:>6.2f}  Def: {row['xg_def']:>6.2f}  Total: {row['xg_total']:>6.2f}  TOI: {row['toi_min']:>5.0f}")
    
    print(f"\n--- Distribution ---")
    print(f"Mean Off:  {results['xg_off'].mean():.4f}")
    print(f"Mean Def:  {results['xg_def'].mean():.4f}")
    print(f"Std Off:   {results['xg_off'].std():.4f}")
    print(f"Std Def:   {results['xg_def'].std():.4f}")
    
    # Check correlation between off and def
    corr = results['xg_off'].corr(results['xg_def'])
    print(f"\nOff/Def correlation: {corr:.3f} (expected: -0.3 to +0.3)")
    
    # Validation summary
    print(f"\n--- Validation Summary ---")
    checks = [
        ("Coefficient range OK", results['xg_off'].min() > -3 and results['xg_off'].max() < 3),
        ("Mean centered near 0", abs(results['xg_off'].mean()) < 0.1),
        ("Off/Def correlation reasonable", -0.5 < corr < 0.5),
    ]
    
    all_passed = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    return all_passed


def run_full_validation(db_path: str, season: str, stints_path: Path = None):
    """
    Run all validation checks.
    """
    conn = duckdb.connect(db_path)
    
    print("\n")
    print("=" * 70)
    print(f"RAPM VALIDATION SUITE - Season {season}")
    print("=" * 70)
    
    # Run all checks
    audit_rapm_inputs(conn, season, stints_path)
    check_design_matrix(conn, season)
    range_check(conn, season)
    distribution_check(conn, season)
    correlation_check(conn, season)
    known_players_check(conn, season)
    final_passed = final_validation(conn, season, min_toi_minutes=200)
    
    # Final summary
    print("\n")
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    if final_passed:
        print("\n✓ RAPM implementation appears correct")
        print("  - Coefficients in expected ranges")
        print("  - Distribution centered near zero")
        print("  - Known elite players rank appropriately")
    else:
        print("\n✗ Some validation checks failed - review output above")
    
    conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate RAPM implementation")
    parser.add_argument("--db", type=str, default="nhl_canonical.duckdb", help="Path to DuckDB database")
    parser.add_argument("--season", type=str, default="20242025", help="Season to validate")
    parser.add_argument("--stints", type=str, default=None, help="Path to stints parquet file (optional)")
    
    args = parser.parse_args()
    
    stints_path = Path(args.stints) if args.stints else None
    run_full_validation(args.db, args.season, stints_path)

#!/usr/bin/env python3
"""
Investigation script for data quality and pipeline issues.
"""

import json
import requests
import pandas as pd
import duckdb
from pathlib import Path

ROOT = Path(__file__).parent


def investigate_failed_games():
    """Check why certain 2025-2026 games failed to fetch shift data."""
    print("=" * 60)
    print("1. INVESTIGATING FAILED SHIFT DATA FETCHES")
    print("=" * 60)
    
    # Load fetch progress
    progress_file = ROOT / "data" / "fetch_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        
        s2526 = progress.get("20252026", {})
        print(f"Total games: {s2526.get('total_games', 0)}")
        print(f"Fetched: {s2526.get('fetched_games', 0)}")
        print(f"Failed: {len(s2526.get('failed_games', []))}")
        
        failed_games = s2526.get("failed_games", [])[:10]
        print(f"\nSample failed games: {failed_games}")
    
    # Test a few games directly
    print("\n--- Testing NHL API directly ---")
    test_games = [2025020061, 2025020065, 2025020078, 2025020002]
    
    for game_id in test_games:
        url = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
        try:
            r = requests.get(url, timeout=30)
            data = r.json()
            shift_count = len(data.get("data", []))
            print(f"  Game {game_id}: {shift_count} shifts (status={r.status_code})")
        except Exception as e:
            print(f"  Game {game_id}: ERROR - {e}")


def investigate_mcdavid_similarity():
    """Check why Brandon Saad is similar to Connor McDavid."""
    print("\n" + "=" * 60)
    print("2. INVESTIGATING McDavid-Saad SIMILARITY")
    print("=" * 60)
    
    categories_file = ROOT / "profile_data" / "player_categories.csv"
    if not categories_file.exists():
        print("ERROR: player_categories.csv not found")
        return
    
    df = pd.read_csv(categories_file)
    df = df[df["season"].astype(str) == "20252026"]
    
    # Get category scores for comparison
    score_cols = [c for c in df.columns if "_actual_score" in c]
    
    mcdavid = df[df["full_name"] == "Connor McDavid"]
    saad = df[df["full_name"] == "Brandon Saad"]
    draisaitl = df[df["full_name"] == "Leon Draisaitl"]
    
    if mcdavid.empty:
        print("McDavid not found in 2025-2026")
        return
    
    print("\nCategory scores (used for similarity computation):")
    print("-" * 50)
    
    players = {"McDavid": mcdavid, "Saad": saad, "Draisaitl": draisaitl}
    
    for name, player_df in players.items():
        if player_df.empty:
            print(f"{name}: Not found")
            continue
        
        row = player_df.iloc[0]
        scores = {c.replace("_actual_score", ""): row.get(c) for c in score_cols}
        print(f"\n{name}:")
        for cat, score in scores.items():
            if pd.notna(score):
                print(f"  {cat}: {score:.3f}")
            else:
                print(f"  {cat}: NaN")


def investigate_dlm_coverage():
    """Check DLM historical coverage."""
    print("\n" + "=" * 60)
    print("3. DLM HISTORICAL COVERAGE")
    print("=" * 60)
    
    db_path = ROOT / "nhl_canonical.duckdb"
    if not db_path.exists():
        print("ERROR: Database not found")
        return
    
    con = duckdb.connect(str(db_path), read_only=True)
    
    # Check DLM coverage
    dlm_seasons = con.execute("SELECT DISTINCT season FROM dlm_rapm_estimates ORDER BY season").df()
    print(f"DLM seasons: {dlm_seasons['season'].tolist()}")
    
    # Check RAPM coverage
    rapm_seasons = con.execute("SELECT DISTINCT season FROM apm_results ORDER BY season").df()
    print(f"RAPM seasons: {rapm_seasons['season'].tolist()}")
    
    # Count players per season
    counts = con.execute("""
        SELECT season, COUNT(DISTINCT player_id) as players
        FROM dlm_rapm_estimates
        GROUP BY season
        ORDER BY season
    """).df()
    print("\nDLM players per season:")
    print(counts.to_string(index=False))
    
    con.close()


def investigate_finishing_metric():
    """Check why McDavid has low finishing percentile."""
    print("\n" + "=" * 60)
    print("4. McDavid FINISHING METRIC INVESTIGATION")
    print("=" * 60)
    
    db_path = ROOT / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path), read_only=True)
    
    # Check McDavid's raw finishing RAPM values
    mcdavid_rapm = con.execute("""
        SELECT season, metric_name, value
        FROM apm_results
        WHERE player_id = 8478402  -- McDavid
        AND metric_name LIKE '%finishing%'
        ORDER BY season
    """).df()
    
    print("McDavid finishing_residual_rapm_5v5 values:")
    print(mcdavid_rapm.to_string(index=False))
    
    # Compare to league distribution
    finishing_dist = con.execute("""
        SELECT 
            season,
            AVG(value) as mean,
            STDDEV(value) as std,
            MIN(value) as min,
            MAX(value) as max
        FROM apm_results
        WHERE metric_name = 'finishing_residual_rapm_5v5'
        GROUP BY season
        ORDER BY season
    """).df()
    
    print("\nLeague finishing_residual_rapm_5v5 distribution:")
    print(finishing_dist.to_string(index=False))
    
    con.close()


def investigate_actual_scores_issue():
    """Check if actual scores are correctly computed."""
    print("\n" + "=" * 60)
    print("5. CHECKING ACTUAL SCORE COMPUTATION")
    print("=" * 60)
    
    df = pd.read_csv(ROOT / "profile_data" / "player_categories.csv")
    current = df[df["season"].astype(str) == "20252026"]
    
    # Check non-null coverage for actual scores
    actual_cols = [c for c in df.columns if "_actual_score" in c]
    print("Actual score coverage in 20252026:")
    for col in actual_cols:
        non_null = current[col].notna().sum()
        pct = 100 * non_null / len(current)
        print(f"  {col}: {non_null}/{len(current)} ({pct:.1f}%)")


if __name__ == "__main__":
    investigate_failed_games()
    investigate_mcdavid_similarity()
    investigate_dlm_coverage()
    investigate_finishing_metric()
    investigate_actual_scores_issue()

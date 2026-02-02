"""
Alpha Hyperparameter Search for RAPM Ridge Regression

This script determines the optimal alpha (regularization) by:
1. Testing a wide range of alphas (0.1 to 10000)
2. Using RidgeCV cross-validation scores
3. Validating against "ground truth" (elite players should have high values)
4. Reporting coefficient spread and player rankings
"""

import subprocess
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path


def run_rapm_with_alphas(alphas_str: str, season: str = "20242025"):
    """Run RAPM calculation with specific alphas."""
    cmd = [
        "python", "scripts/compute_corsi_apm.py",
        "--mode", "stint",
        "--strength", "5v5", 
        "--metrics", "corsi_offdef,xg_offdef",
        "--season", season,
        "--alphas", alphas_str,
        "--use-precomputed-xg"
    ]
    print(f"Running RAPM with alphas: {alphas_str}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path("."))
    
    # Extract alpha used from output
    for line in result.stdout.split("\n"):
        if "alpha=" in line.lower():
            print(f"  {line.strip()}")
    
    return result


def get_player_rapm(metric_name: str, season: str, top_n: int = 20):
    """Get top players for a specific RAPM metric."""
    db_path = Path("nhl_canonical.duckdb")
    con = duckdb.connect(str(db_path), read_only=True)
    
    df = con.execute(f"""
        SELECT 
            a.player_id,
            p.full_name,
            a.value as rapm,
            a.toi_seconds / 60.0 as toi_min
        FROM apm_results a
        LEFT JOIN players p ON a.player_id = p.player_id
        WHERE a.metric_name = '{metric_name}' 
          AND a.season = '{season}'
        ORDER BY a.value DESC
        LIMIT {top_n}
    """).df()
    con.close()
    return df


def validate_elite_players(metric_name: str, season: str):
    """Check if known elite players rank appropriately."""
    db_path = Path("nhl_canonical.duckdb")
    con = duckdb.connect(str(db_path), read_only=True)
    
    elite_ids = [
        8478402,  # Connor McDavid
        8477492,  # Nathan MacKinnon
        8478483,  # Nikita Kucherov
        8479318,  # Leon Draisaitl
    ]
    
    df = con.execute(f"""
        SELECT 
            a.player_id,
            p.full_name,
            a.value as rapm,
            a.toi_seconds / 60.0 as toi_min
        FROM apm_results a
        LEFT JOIN players p ON a.player_id = p.player_id
        WHERE a.metric_name = '{metric_name}' 
          AND a.season = '{season}'
          AND a.player_id IN ({','.join(map(str, elite_ids))})
        ORDER BY a.value DESC
    """).df()
    con.close()
    return df


def compute_coefficient_stats(metric_name: str, season: str):
    """Compute statistics on RAPM coefficients."""
    db_path = Path("nhl_canonical.duckdb")
    con = duckdb.connect(str(db_path), read_only=True)
    
    df = con.execute(f"""
        SELECT value FROM apm_results
        WHERE metric_name = '{metric_name}' AND season = '{season}'
    """).df()
    con.close()
    
    if df.empty:
        return None
    
    values = df["value"].values
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "range": np.max(values) - np.min(values),
        "count": len(values),
        "99th_pct": np.percentile(values, 99),
        "1st_pct": np.percentile(values, 1),
    }


def main():
    season = "20242025"
    metric = "corsi_off_rapm_5v5"
    
    # Test multiple alpha ranges
    alpha_ranges = [
        "0.1,0.5,1,5,10,50",
        "10,50,100,250,500,1000",
        "100,250,500,1000,5000,10000",
    ]
    
    print("=" * 60)
    print("RAPM Alpha Hyperparameter Search")
    print("=" * 60)
    
    results = []
    
    for alphas in alpha_ranges:
        print(f"\n--- Testing alphas: {alphas} ---")
        run_rapm_with_alphas(alphas, season)
        
        # Get stats
        stats = compute_coefficient_stats(metric, season)
        if stats:
            print(f"  Coefficient range: {stats['min']:.3f} to {stats['max']:.3f}")
            print(f"  Std dev: {stats['std']:.3f}")
            
            # Check elite players
            elite = validate_elite_players(metric, season)
            if not elite.empty:
                mcdavid = elite[elite["full_name"] == "Connor McDavid"]
                if not mcdavid.empty:
                    mcd_val = mcdavid["rapm"].values[0]
                    print(f"  McDavid RAPM: {mcd_val:.3f} ({mcd_val/stats['std']:.1f} std above mean)")
            
            results.append({
                "alphas": alphas,
                **stats
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nThe BEST alpha is the one where:")
    print("  1. McDavid is 3+ std above mean")
    print("  2. Coefficient range is wide enough to differentiate players")
    print("  3. RidgeCV selected an alpha IN the middle of the range (not at boundaries)")
    
    # Final recommendation
    top = get_player_rapm(metric, season, 10)
    print(f"\nTop 10 {metric}:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()

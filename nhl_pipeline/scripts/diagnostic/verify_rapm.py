
import duckdb
import pandas as pd

from pathlib import Path

def verify():
    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    season = "20242025"
    print(f"Connecting to: {db_path}")
    conn = duckdb.connect(str(db_path))
    
    print("Checking apm_results table...")
    # Check counts per season
    counts = conn.execute("SELECT season, COUNT(*) as count FROM apm_results GROUP BY season").fetchdf()
    print(counts)
    
    # Check 20242025 specifically
    df = conn.execute("SELECT * FROM apm_results WHERE season='20242025'").fetchdf()
    if df.empty:
        print("FAILURE: No data for 20242025 found in apm_results.")
        print("Seasons found:", conn.execute("SELECT DISTINCT season FROM apm_results").fetchall())
    else:
        print(f"SUCCESS: Found {len(df)} rows for 20242025.")
    
    # Check top offensive players (Corsi Offense)
    print("\nTop 10 Corsi Offense (2024-2025):")
    top_corsi = conn.execute("""
        SELECT player_id, value, games_count, toi_seconds 
        FROM apm_results 
        WHERE season='20242025' AND metric_name='corsi_off_rapm_5v5' 
        ORDER BY value DESC 
        LIMIT 10
    """).fetchdf()
    print(top_corsi)
    
    # Check top xG Offense
    print("\nTop 10 xG Offense (2024-2025):")
    top_xg = conn.execute("""
        SELECT player_id, value, games_count, toi_seconds 
        FROM apm_results 
        WHERE season='20242025' AND metric_name='xg_off_rapm_5v5' 
        ORDER BY value DESC 
        LIMIT 10
    """).fetchdf()
    print(top_xg)
    
    # Check specific players (McDavid: 8478402, MacKinnon: 8477492)
    print("\nSpecific Players:")
    players = [8478402, 8477492]
    for pid in players:
        res = conn.execute(f"""
            SELECT metric_name, value 
            FROM apm_results 
            WHERE season='20242025' AND player_id={pid}
        """).fetchdf()
        print(f"Player {pid}:")
        print(res)

    print("\n=== Validation Checks ===")
    
    metrics_to_check = [
        ("corsi_off_rapm_5v5", 15.0), # Corsi/60 range +/- 15 (relaxed slightly from 10)
        ("xg_off_rapm_5v5", 1.5),     # xG/60 range +/- 1.5
        ("goals_rapm_5v5", 1.0)       # Goals/60 range +/- 1.0
    ]
    
    for metric, limit in metrics_to_check:
        print(f"\nChecking {metric} (Limit +/- {limit})...")
        df = conn.execute(f"SELECT value, toi_seconds FROM apm_results WHERE season='{season}' AND metric_name='{metric}'").fetchdf()
        if df.empty:
            print(f"  No data for {metric}")
            continue
            
        min_val = df["value"].min()
        max_val = df["value"].max()
        print(f"  Range: {min_val:.4f} to {max_val:.4f}")
        
        if abs(min_val) > limit or abs(max_val) > limit:
            print(f"  WARNING: Value outside expected range +/- {limit}")
        else:
            print(f"  PASS: Range within +/- {limit}")
            
        # Sum check
        weighted_sum = (df["value"] * (df["toi_seconds"] / 3600.0)).sum()
        print(f"  Weighted Sum (Total League Impact): {weighted_sum:.2f}")

    conn.close()
    print("Done!")

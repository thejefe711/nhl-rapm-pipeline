import duckdb
import pandas as pd

def rapm_sum_check(conn, season: str):
    """
    Total offensive impact × TOI - total defensive impact × TOI should ≈ 0
    """
    
    # Load xG RAPM results
    try:
        off = conn.execute(f"""
            SELECT player_id, value as off_rapm, toi_seconds 
            FROM apm_results 
            WHERE season = '{season}' AND metric_name = 'xg_off_rapm_5v5'
        """).df()
        
        def_ = conn.execute(f"""
            SELECT player_id, value as def_rapm
            FROM apm_results 
            WHERE season = '{season}' AND metric_name = 'xg_def_rapm_5v5'
        """).df()
    except Exception as e:
        print(f"Error querying DuckDB: {e}")
        return

    if off.empty or def_.empty:
        print("No RAPM data found for season", season)
        return

    merged = off.merge(def_, on="player_id")
    merged["toi_hours"] = merged["toi_seconds"] / 3600
    
    total_off = (merged["off_rapm"] * merged["toi_hours"]).sum()
    total_def = (merged["def_rapm"] * merged["toi_hours"]).sum()
    net = total_off - total_def
    
    print("=== SUM CHECK ===")
    print(f"Total offensive impact × TOI: {total_off:.2f}")
    print(f"Total defensive impact × TOI: {total_def:.2f}")
    print(f"Net (should be ≈ 0): {net:.2f}")
    
    if abs(net) < 50:
        print("✓ PASS")
    else:
        print("✗ FAIL - Check for systematic bias")

if __name__ == "__main__":
    # Run it
    conn = duckdb.connect("nhl_pipeline/nhl_canonical.duckdb")
    rapm_sum_check(conn, "20242025")

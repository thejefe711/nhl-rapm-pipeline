import duckdb
import pandas as pd

def verify_rapm_outputs(season: int):
    conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
    
    print(f"\n=== RAPM OUTPUT VERIFICATION (Season {season}) ===")
    
    # Fetch results with TOI
    query = """
    SELECT 
        p.full_name as player_name,
        r.metric_name,
        r.value,
        r.toi_seconds / 60.0 as toi_minutes
    FROM apm_results r
    JOIN players p ON r.player_id = p.player_id
    WHERE (r.season = ? OR r.season = ?) 
    AND r.metric_name IN ('xg_off_rapm_5v5', 'xg_def_rapm_5v5')
    """
    
    season_long = f"{season}{season+1}"
    df = conn.execute(query, [str(season), season_long]).fetchdf()
    
    if df.empty:
        print(f"No RAPM results found for season {season}")
        return

    # Pivot for easier handling
    df_pivot = df.pivot_table(
        index=['player_name', 'toi_minutes'], 
        columns='metric_name', 
        values='value'
    ).reset_index()
    
    if 'xg_off_rapm_5v5' not in df_pivot.columns:
        print("Missing xG RAPM metrics in results.")
        return

    # 1. Sum Check
    # Impact = rate * (toi_minutes / 60) -> wait, rate is per 60, so impact = rate * (toi_minutes / 60)
    # Actually, value is per 60. Total impact = value * (toi_minutes / 60).
    
    total_off = (df_pivot['xg_off_rapm_5v5'] * (df_pivot['toi_minutes'] / 60)).sum()
    total_def = (df_pivot['xg_def_rapm_5v5'] * (df_pivot['toi_minutes'] / 60)).sum()
    net = total_off - total_def # Should be close to 0 if centered? 
    
    print(f"\nSum Check:")
    print(f"  Total Offensive Impact: {total_off:.2f}")
    print(f"  Total Defensive Impact: {total_def:.2f}")
    print(f"  Net (Off + Def): {total_off + total_def:.2f} (Note: Sign convention varies)")
    
    # 2. Top/Bottom 10
    print("\nTop 10 Offensive xG RAPM:")
    print(df_pivot.nlargest(10, 'xg_off_rapm_5v5')[['player_name', 'xg_off_rapm_5v5', 'toi_minutes']].to_string(index=False))

    print("\nBottom 10 Offensive xG RAPM:")
    print(df_pivot.nsmallest(10, 'xg_off_rapm_5v5')[['player_name', 'xg_off_rapm_5v5', 'toi_minutes']].to_string(index=False))

    print("\nTop 10 Defensive xG RAPM (Best Suppression):")
    print(df_pivot.nlargest(10, 'xg_def_rapm_5v5')[['player_name', 'xg_def_rapm_5v5', 'toi_minutes']].to_string(index=False))
    
    print("\nBottom 10 Defensive xG RAPM (Worst Suppression):")
    print(df_pivot.nsmallest(10, 'xg_def_rapm_5v5')[['player_name', 'xg_def_rapm_5v5', 'toi_minutes']].to_string(index=False))

if __name__ == "__main__":
    verify_rapm_outputs(2024)

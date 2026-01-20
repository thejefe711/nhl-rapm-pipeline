#!/usr/bin/env python3
"""Quick status check for rolling embeddings computation."""

import duckdb
from pathlib import Path

db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
con = duckdb.connect(str(db_path))

# Check windows completed per season
df = con.execute("""
    SELECT 
        season,
        COUNT(DISTINCT window_end_game_id) as windows_completed,
        MIN(window_end_game_id) as first_window,
        MAX(window_end_game_id) as last_window
    FROM rolling_latent_skills
    WHERE model_name = 'sae_apm_v1_k12_a1'
    GROUP BY season
    ORDER BY season DESC
""").df()

print("=" * 60)
print("ROLLING EMBEDDINGS STATUS")
print("=" * 60)
if df.empty:
    print("No windows completed yet.")
else:
    print(df.to_string(index=False))
    
    # For 20242025, show more detail
    if "20242025" in df["season"].values:
        detail = con.execute("""
            SELECT 
                window_end_game_id,
                window_end_time_utc,
                COUNT(DISTINCT player_id) as players,
                COUNT(*) as total_rows
            FROM rolling_latent_skills
            WHERE model_name = 'sae_apm_v1_k12_a1' AND season = '20242025'
            GROUP BY window_end_game_id, window_end_time_utc
            ORDER BY window_end_game_id
            LIMIT 5
        """).df()
        print("\n--- Latest 5 windows (20242025) ---")
        print(detail.to_string(index=False))

con.close()

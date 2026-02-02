import duckdb
import pandas as pd
from pathlib import Path

db_path = Path("nhl_pipeline/nhl_canonical.duckdb")
conn = duckdb.connect(str(db_path))

# Find McDavid's ID
mcdavid = conn.execute("SELECT player_id, full_name FROM players WHERE full_name LIKE '%McDavid%'").df()
print("Found Player:")
print(mcdavid)

if not mcdavid.empty:
    pid = mcdavid.iloc[0]['player_id']
    
    # Query metrics across seasons
    # We'll focus on corsi_rapm_5v5 and xg_rapm_5v5
    query = f"""
    SELECT 
        season, 
        metric_name, 
        value
    FROM apm_results 
    WHERE player_id = {pid}
    AND metric_name IN ('corsi_rapm_5v5', 'xg_rapm_5v5', 'goals_rapm_5v5')
    ORDER BY season, metric_name
    """
    results = conn.execute(query).df()
    
    # Pivot for better readability
    if not results.empty:
        pivot_df = results.pivot(index='season', columns='metric_name', values='value')
        print("\nConnor McDavid RAPM Throughout the Years:")
        print("Season | Corsi RAPM 5v5 | xG RAPM 5v5 | Goals RAPM 5v5")
        print("-" * 60)
        for season, row in pivot_df.iterrows():
            corsi = row.get('corsi_rapm_5v5', 0)
            xg = row.get('xg_rapm_5v5', 0)
            goals = row.get('goals_rapm_5v5', 0)
            print(f"{season} | {corsi:14.4f} | {xg:11.4f} | {goals:14.4f}")
    else:
        print("\nNo RAPM results found for McDavid.")

conn.close()

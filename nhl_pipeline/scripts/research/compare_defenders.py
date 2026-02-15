import duckdb
import pandas as pd
from pathlib import Path

db_path = Path("nhl_pipeline/nhl_canonical.duckdb")
conn = duckdb.connect(str(db_path))

players_to_check = [
    "Connor McDavid",
    "Jaccob Slavin",
    "Chris Tanev",
    "Adam Pelech"
]

# Get IDs
player_info = conn.execute(f"SELECT player_id, full_name FROM players WHERE full_name IN {tuple(players_to_check)}").df()
print("Player Info:")
print(player_info)

# Get defensive metrics for 2023-2024 and 2024-2025
seasons = ['20232024', '20242025']
metrics = [
    'blocked_shot_to_xg_swing_rapm_5v5_w10',
    'takeaway_to_xg_swing_rapm_5v5_w10',
    'turnover_to_xg_swing_rapm_5v5_w10',
    'penalties_taken_rapm_5v5'
]

results = []
for _, row in player_info.iterrows():
    pid = row['player_id']
    name = row['full_name']
    for season in seasons:
        for metric in metrics:
            res = conn.execute(f"SELECT value, events_count FROM apm_results WHERE player_id = {pid} AND season = '{season}' AND metric_name = '{metric}'").fetchone()
            if res:
                results.append({
                    "Name": name,
                    "Season": season,
                    "Metric": metric,
                    "Value": res[0],
                    "Count": res[1]
                })

df = pd.DataFrame(results)
if not df.empty:
    print("\nDefensive Action Comparison (Value and Raw Event Count):")
    header = f"{'Name':<15} | {'Season':<10} | {'Blocks (Val/Ct)':<16} | {'Takeaways (Val/Ct)':<18} | {'Turnovers (Val/Ct)':<18}"
    print(header)
    print("-" * len(header))
    for name in players_to_check:
        for season in seasons:
            row = df[(df['Name'] == name) & (df['Season'] == season)]
            if not row.empty:
                vals = {}
                counts = {}
                for m in metrics:
                    m_row = row[row['Metric'] == m]
                    vals[m] = m_row['Value'].values[0] if not m_row.empty else 0
                    counts[m] = m_row['Count'].values[0] if not m_row.empty else 0
                
                print(f"{name:<15} | {season:<10} | {vals[metrics[0]]:7.4f}/{counts[metrics[0]]:<4.0f} | {vals[metrics[1]]:8.4f}/{counts[metrics[1]]:<4.0f} | {vals[metrics[2]]:8.4f}/{counts[metrics[2]]:<4.0f}")

conn.close()

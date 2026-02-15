import duckdb
import pandas as pd

conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')

# Check schema
print("Players table schema:")
print(conn.execute("DESCRIBE players").df())

# Check available seasons
print("\nAvailable seasons in apm_results:")
print(conn.execute("SELECT DISTINCT season FROM apm_results").df())

# Query for top players in blocked_shot_to_xg_swing_rapm_5v5_w10 for 2023-2024
# Lowering TOI threshold to 200 minutes (12000 seconds)
query = """
SELECT 
    p.full_name,
    a.value as block_xg_swing,
    a.toi_seconds / 60.0 as toi_minutes
FROM apm_results a
JOIN players p ON a.player_id = p.player_id
WHERE a.season = '20232024'
  AND a.metric_name = 'blocked_shot_to_xg_swing_rapm_5v5_w10'
  AND a.toi_seconds >= 12000
ORDER BY a.value DESC
LIMIT 20
"""

res = conn.execute(query).df()
print("Top 20 Players for Block xG Swing RAPM (2023-2024, Min 500 mins):")
print(res.to_string(index=False))

with open('top_block_rapm.txt', 'w') as f:
    f.write(res.to_string(index=False))

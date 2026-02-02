import duckdb
import pandas as pd

conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
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
for _, row in res.iterrows():
    print(f"{row.full_name}|{row.block_xg_swing:.4f}|{row.toi_minutes:.1f}")

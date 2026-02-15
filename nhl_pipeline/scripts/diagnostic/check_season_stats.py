import duckdb
import pandas as pd

conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
query = """
SELECT 
    season, 
    MAX(games_count) as games, 
    CAST(SUM(toi_seconds) / 600.0 AS INTEGER) as total_toi_min
FROM apm_results 
WHERE metric_name = 'xg_off_rapm_5v5'
GROUP BY season 
ORDER BY season
"""
df = conn.execute(query).df()
print(df)
conn.close()

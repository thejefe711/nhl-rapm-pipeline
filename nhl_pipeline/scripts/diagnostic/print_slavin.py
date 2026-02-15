import duckdb
import pandas as pd

conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
slavin_id = 8476958
season = '20232024'

query = f"""
WITH metric_ranks AS (
    SELECT 
        season,
        metric_name,
        player_id,
        value,
        RANK() OVER (PARTITION BY season, metric_name ORDER BY value DESC) as rank_desc,
        RANK() OVER (PARTITION BY season, metric_name ORDER BY value ASC) as rank_asc,
        COUNT(*) OVER (PARTITION BY season, metric_name) as total_players
    FROM apm_results
    WHERE toi_seconds >= 30000
)
SELECT 
    metric_name,
    value,
    CASE 
        WHEN metric_name LIKE '%def%' OR metric_name LIKE '%pk%' OR metric_name LIKE '%taken%' THEN rank_asc
        ELSE rank_desc
    END as final_rank,
    total_players
FROM metric_ranks
WHERE player_id = {slavin_id}
  AND season = '{season}'
ORDER BY metric_name
"""

res = conn.execute(query).df()
for _, row in res.iterrows():
    print(f"{row.metric_name}|{row.value:.4f}|{row.final_rank}|{row.total_players}")

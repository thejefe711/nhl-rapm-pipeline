import duckdb
import pandas as pd

conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')

# 1. Get all metric names
res = conn.execute("SELECT DISTINCT metric_name FROM apm_results").df()

# 2. Get specific player values
players = [8478402, 8476958, 8476917]
query = f"""
SELECT 
    season,
    player_id,
    metric_name,
    value
FROM apm_results
WHERE player_id IN ({','.join(map(str, players))})
  AND (metric_name LIKE '%block%' OR metric_name LIKE '%shot%')
ORDER BY season, metric_name, player_id
"""
res_players = conn.execute(query).df()

# 3. Write to file
with open('rapm_metrics_results.txt', 'w') as f:
    f.write("All available RAPM metrics:\n")
    for m in sorted(res['metric_name'].tolist()):
        f.write(f"- {m}\n")
    
    f.write("\nBlock/Shot RAPM for McDavid, Slavin, Pelech:\n")
    f.write(res_players.to_string(index=False))

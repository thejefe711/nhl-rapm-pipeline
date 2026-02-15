import duckdb
import pandas as pd

conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
season = '20232024'

res = conn.execute(f"SELECT metric_name, COUNT(*) as player_count, AVG(toi_seconds) as avg_toi FROM apm_results WHERE season = '{season}' GROUP BY 1").df()
for _, row in res.iterrows():
    print(f"{row.metric_name}|{row.player_count}|{row.avg_toi}")

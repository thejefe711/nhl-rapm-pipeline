import duckdb
import pandas as pd

conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
season = '20232024'

res = conn.execute(f"SELECT metric_name, COUNT(*) as player_count, AVG(toi_seconds) as avg_toi FROM apm_results WHERE season = '{season}' GROUP BY 1").df()
print(res.to_string(index=False))

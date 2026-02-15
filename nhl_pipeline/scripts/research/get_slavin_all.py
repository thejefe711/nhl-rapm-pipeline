import duckdb
import pandas as pd

conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
slavin_id = 8476958
season = '20232024'

res = conn.execute(f"SELECT metric_name, value, toi_seconds FROM apm_results WHERE player_id = {slavin_id} AND season = '{season}'").df()
print(res.to_string(index=False))

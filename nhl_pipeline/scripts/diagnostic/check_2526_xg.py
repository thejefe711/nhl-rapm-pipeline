import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
q = "SELECT metric_name, COUNT(*), MIN(created_at), MAX(created_at) FROM apm_results WHERE season = '20252026' AND metric_name = 'xg_rapm_5v5' GROUP BY 1"
print(con.execute(q).df())
con.close()

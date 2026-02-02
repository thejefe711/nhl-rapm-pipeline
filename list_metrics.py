import duckdb
from pathlib import Path

db_path = Path("nhl_pipeline/nhl_canonical.duckdb")
conn = duckdb.connect(str(db_path))

res = conn.execute("SELECT DISTINCT metric_name FROM apm_results ORDER BY 1").fetchall()
for r in res:
    print(r[0])

conn.close()

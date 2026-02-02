import duckdb
import pandas as pd
from pathlib import Path

db_path = Path("nhl_pipeline/nhl_canonical.duckdb")
if not db_path.exists():
    print(f"Database not found at {db_path}")
else:
    conn = duckdb.connect(str(db_path))
    try:
        df = conn.execute("SELECT season, count(distinct metric_name) as metrics, count(*) as total_rows FROM apm_results GROUP BY 1 ORDER BY 1").df()
        print(df)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

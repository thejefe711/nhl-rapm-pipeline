import duckdb
import pandas as pd
from pathlib import Path

db_path = Path("nhl_pipeline/nhl_canonical.duckdb")
conn = duckdb.connect(str(db_path))
try:
    # Get games_count per season from apm_results
    # It's stored in the games_count column for every row
    query = """
    SELECT season, MAX(games_count) as games_processed
    FROM apm_results
    GROUP BY season
    ORDER BY season
    """
    df = conn.execute(query).df()
    print(df)
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()

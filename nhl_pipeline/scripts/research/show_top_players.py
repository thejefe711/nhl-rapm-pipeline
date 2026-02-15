import duckdb
import pandas as pd
from pathlib import Path

db_path = Path("nhl_pipeline/nhl_canonical.duckdb")
conn = duckdb.connect(str(db_path))
try:
    # Get top 10 players by Corsi RAPM for 2023-2024
    query = """
    SELECT player_id, value, toi_seconds / 60 as toi_min
    FROM apm_results
    WHERE season = '20232024' AND metric_name = 'corsi_rapm_5v5'
    ORDER BY value DESC
    LIMIT 10
    """
    df = conn.execute(query).df()
    print("Top 10 Players (Corsi RAPM 5v5) - 2023-2024")
    print(df)
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()

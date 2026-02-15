import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
q = """
SELECT 
    shot_type,
    COUNT(*) as count
FROM events
WHERE LEFT(CAST(game_id AS VARCHAR), 4) = '2025'
AND event_type IN ('SHOT', 'MISSED_SHOT', 'GOAL')
GROUP BY 1;
"""
print(con.execute(q).df())
con.close()

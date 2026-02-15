import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
q = """
SELECT 
    p.full_name,
    COUNT(DISTINCT e.game_id) as games_in_events
FROM events e
JOIN players p ON e.player_1_id = p.player_id
WHERE p.full_name IN ('Adam Fox', 'Quinn Hughes', 'Cale Makar')
AND LEFT(CAST(e.game_id AS VARCHAR), 4) = '2024'
GROUP BY 1;
"""
print(con.execute(q).df())
con.close()

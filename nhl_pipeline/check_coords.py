import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

print('--- Check coordinates by season in events ---')
q = """
SELECT 
    LEFT(CAST(game_id AS VARCHAR), 4) as season_prefix,
    COUNT(*) as total_events,
    SUM(CASE WHEN x_coord IS NOT NULL AND x_coord != 0 THEN 1 ELSE 0 END) as has_x,
    SUM(CASE WHEN y_coord IS NOT NULL AND y_coord != 0 THEN 1 ELSE 0 END) as has_y
FROM events
GROUP BY 1
ORDER BY 1;
"""
print(con.execute(q).df().to_string(index=False))

con.close()

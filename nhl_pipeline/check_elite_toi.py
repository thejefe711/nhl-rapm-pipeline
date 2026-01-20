import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
q = """
SELECT 
    p.full_name,
    a.player_id,
    a.season,
    MAX(CASE WHEN a.metric_name = 'corsi_off_rapm_5v5' THEN a.value ELSE NULL END) as corsi_off,
    MAX(CASE WHEN a.metric_name = 'xg_off_rapm_5v5' THEN a.value ELSE NULL END) as xg_off,
    MAX(a.toi_seconds) / 60.0 as toi_min
FROM apm_results a
JOIN players p ON a.player_id = p.player_id
WHERE a.season = '20242025'
AND p.full_name IN ('Adam Fox', 'Quinn Hughes', 'Cale Makar')
GROUP BY p.full_name, a.player_id, a.season
ORDER BY p.full_name;
"""
print(con.execute(q).df())
con.close()

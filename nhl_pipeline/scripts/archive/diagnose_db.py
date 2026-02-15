import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

print('--- Check xG by season in apm_results ---')
q1 = """
SELECT 
    season,
    COUNT(*) as total_rows,
    SUM(CASE WHEN metric_name = 'xg_off_rapm_5v5' AND value = 0 THEN 1 ELSE 0 END) as zero_xg_count,
    AVG(CASE WHEN metric_name = 'xg_off_rapm_5v5' THEN value ELSE NULL END) as avg_xg_off
FROM apm_results
GROUP BY season
ORDER BY season;
"""
print(con.execute(q1).df().to_string(index=False))

print('\n--- Check for xG column in all tables ---')
tables = con.execute("SHOW TABLES").df()['name'].tolist()
for table in tables:
    cols = con.execute(f"PRAGMA table_info('{table}')").df()['name'].tolist()
    if 'xg' in [c.lower() for c in cols]:
        print(f"Found 'xg' in table: {table}")
        q = f"""
        SELECT 
            LEFT(CAST(game_id AS VARCHAR), 4) as season_prefix,
            COUNT(*) as total_rows,
            SUM(CASE WHEN xg IS NOT NULL AND xg > 0 THEN 1 ELSE 0 END) as has_xg
        FROM {table}
        GROUP BY 1
        ORDER BY 1;
        """
        try:
            print(con.execute(q).df().to_string(index=False))
        except Exception as e:
            print(f"Error querying {table}: {e}")

print('\n--- Check top offensive D in 2024-2025 ---')
q3 = """
SELECT 
    p.full_name,
    a.player_id,
    a.season,
    MAX(CASE WHEN a.metric_name = 'corsi_off_rapm_5v5' THEN a.value ELSE NULL END) as corsi_off,
    MAX(CASE WHEN a.metric_name = 'xg_off_rapm_5v5' THEN a.value ELSE NULL END) as xg_off
FROM apm_results a
JOIN players p ON a.player_id = p.player_id
WHERE a.season = '20242025'
GROUP BY p.full_name, a.player_id, a.season
ORDER BY corsi_off DESC
LIMIT 20;
"""
print(con.execute(q3).df().to_string(index=False))

print('\n--- Check player joins ---')
q4 = """
SELECT 
    a.player_id,
    p.full_name,
    COUNT(*) as row_count
FROM apm_results a
LEFT JOIN players p ON a.player_id = p.player_id
GROUP BY a.player_id, p.full_name
HAVING p.full_name IS NULL
LIMIT 20;
"""
print(con.execute(q4).df().to_string(index=False))

con.close()

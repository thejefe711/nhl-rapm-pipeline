import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

players = ['Quinn Hughes', 'Cale Makar']
seasons = ['20232024', '20242025']

for name in players:
    print(f"\n=== {name} ===")
    res = con.execute(f"""
        SELECT a.season, a.metric_name, a.value
        FROM apm_results a
        JOIN players p ON a.player_id = p.player_id
        WHERE p.full_name = '{name}'
        AND a.season IN ({','.join(seasons)})
        AND (a.metric_name LIKE '%off_rapm_5v5' OR a.metric_name LIKE '%def_rapm_5v5')
        ORDER BY a.season, a.metric_name
    """).df()
    if res.empty:
        print("No data found in apm_results")
    else:
        print(res.to_string(index=False))

con.close()

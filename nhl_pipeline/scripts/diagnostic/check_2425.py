import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

players = ['Quinn Hughes', 'Cale Makar']
season = '20242025'

for name in players:
    print(f"\n=== {name} ({season}) ===")
    res = con.execute(f"""
        SELECT a.metric_name, a.value
        FROM apm_results a
        JOIN players p ON a.player_id = p.player_id
        WHERE p.full_name = '{name}'
        AND a.season = '{season}'
        ORDER BY a.metric_name
    """).df()
    if res.empty:
        print("No data found in apm_results for this season")
    else:
        print(res.to_string(index=False))

con.close()

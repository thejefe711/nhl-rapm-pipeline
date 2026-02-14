import duckdb
con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
res = con.execute("SELECT season, metric_name, games_count FROM apm_results WHERE games_count = 53 LIMIT 20").fetchall()
for s, m, g in res:
    print(f"{s} | {m} | {g}")
con.close()

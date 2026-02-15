import duckdb
con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
res = con.execute("SELECT season, AVG(games_count) FROM apm_results GROUP BY season ORDER BY season DESC").fetchall()
for s, a in res:
    print(f"{s}: {a}")
con.close()

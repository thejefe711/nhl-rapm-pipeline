import duckdb
con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
res = con.execute("SELECT season, MIN(games_count), MAX(games_count) FROM apm_results GROUP BY season ORDER BY season DESC").fetchall()
for s, mi, ma in res:
    print(f"Season {s}: Games Range {mi} - {ma}")
con.close()

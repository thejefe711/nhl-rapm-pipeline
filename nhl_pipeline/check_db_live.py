import duckdb
con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
res = con.execute("SELECT metric_name, AVG(games_count), COUNT(*) FROM apm_results WHERE season = '20252026' GROUP BY metric_name").fetchall()
for m, g, c in res:
    print(f"{m:<40} | Games: {g:>5.1f} | Rows: {c}")
con.close()

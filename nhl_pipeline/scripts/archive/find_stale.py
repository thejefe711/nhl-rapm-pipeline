import duckdb

con = duckdb.connect("nhl_canonical.duckdb", read_only=True)

with open("stale_metrics.txt", "w") as f:
    for season in ['20252026', '20242025', '20232024', '20222023', '20212022', '20202021']:
        f.write(f"\n=== {season} ===\n")
        res = con.execute("""
            SELECT metric_name, games_count, COUNT(*) as player_count
            FROM apm_results 
            WHERE season = ? AND games_count <= 53
            GROUP BY metric_name, games_count
            ORDER BY metric_name
        """, [season]).fetchall()
        for m, g, c in res:
            f.write(f"  {m:<50} | games={g} | {c} players\n")
        if not res:
            f.write("  (none - all clean)\n")

con.close()
print("Done - see stale_metrics.txt")

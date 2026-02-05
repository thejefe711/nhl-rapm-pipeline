import duckdb

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

with open('games_check_output.txt', 'w') as f:
    f.write("GAMES COUNT CHECK\n")
    f.write("=" * 50 + "\n")

    result = con.execute("""
        SELECT season, 
               MAX(games_count) as games_used,
               COUNT(DISTINCT player_id) as players
        FROM apm_results 
        WHERE metric_name = 'corsi_rapm_5v5'
        GROUP BY season
        ORDER BY season DESC
    """).fetchall()

    f.write("Season     | Games Used | Players\n")
    f.write("-" * 40 + "\n")
    for s, g, p in result:
        f.write(f"{s} | {g:10} | {p}\n")

    f.write("\n")
    f.write("EVENTS COUNT FOR 2025-26\n")
    f.write("=" * 50 + "\n")

    result2 = con.execute("""
        SELECT metric_name, 
               MAX(events_count) as events,
               MAX(games_count) as games
        FROM apm_results 
        WHERE season = '20252026'
        GROUP BY metric_name
        ORDER BY metric_name
    """).fetchall()

    for m, e, g in result2:
        f.write(f"{m}: {e} events, {g} games\n")

con.close()
print("Output written to games_check_output.txt")

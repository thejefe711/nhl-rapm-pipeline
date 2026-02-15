import duckdb

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

with open('all_seasons_check.txt', 'w', encoding='utf-8') as f:
    f.write("GAMES COUNT CONSISTENCY CHECK - ALL SEASONS\n")
    f.write("=" * 70 + "\n\n")
    
    seasons = ['20252026', '20242025', '20232024', '20222023', '20212022', '20202021']
    
    for season in seasons:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"SEASON: {season}\n")
        f.write(f"{'=' * 70}\n")
        
        result = con.execute(f"""
            SELECT metric_name, 
                   MAX(events_count) as events,
                   MAX(games_count) as games
            FROM apm_results 
            WHERE season = '{season}'
            GROUP BY metric_name
            ORDER BY games DESC, metric_name
        """).fetchall()
        
        if not result:
            f.write("  No data found\n")
            continue
        
        # Check for inconsistency
        games_values = set(g for m, e, g in result)
        if len(games_values) > 1:
            f.write(f"  [WARNING] INCONSISTENT GAMES COUNT: {sorted(games_values, reverse=True)}\n\n")
        else:
            f.write(f"  [OK] Consistent: {list(games_values)[0]} games for all metrics\n\n")
        
        f.write(f"  {'Metric':<50} | Events | Games\n")
        f.write(f"  {'-' * 50}-+--------+------\n")
        for m, e, g in result:
            f.write(f"  {m:<50} | {e:>6} | {g:>4}\n")

con.close()
print("Output written to all_seasons_check.txt")

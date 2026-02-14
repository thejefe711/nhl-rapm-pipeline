import duckdb

con = duckdb.connect("nhl_canonical.duckdb", read_only=True)

seasons = ['20252026', '20242025', '20232024', '20222023', '20212022', '20202021']

print(f"{'Season':<12} | {'Metrics':<8} | {'Min Games':<10} | {'Max Games':<10} | {'Sample Metric Check'}")
print("-" * 80)

for season in seasons:
    res = con.execute("""
        SELECT COUNT(DISTINCT metric_name), MIN(games_count), MAX(games_count)
        FROM apm_results WHERE season = ?
    """, [season]).fetchone()
    metrics, min_g, max_g = res
    
    # Check if turnover metrics exist (they were the ones missing before)
    turnover = con.execute("""
        SELECT COUNT(*) FROM apm_results 
        WHERE season = ? AND metric_name LIKE '%turnover%'
    """, [season]).fetchone()[0]
    
    status = "OK" if turnover > 0 and min_g == max_g and min_g > 53 else "NEEDS FIX"
    print(f"{season:<12} | {metrics:<8} | {min_g:>10} | {max_g:>10} | turnover rows: {turnover} [{status}]")

con.close()

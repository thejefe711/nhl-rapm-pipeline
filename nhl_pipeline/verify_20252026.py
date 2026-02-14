"""
Check specific metrics for 2025-26 after full re-run
"""
import duckdb

con = duckdb.connect("nhl_canonical.duckdb", read_only=True)

print("=" * 70)
print("2025-26 SEASON METRIC VERIFICATION")
print("=" * 70)

# Check specific metrics we were worried about
result = con.execute("""
    SELECT metric_name, 
           COUNT(*) as players,
           MAX(games_count) as max_games,
           MIN(games_count) as min_games,
           STDDEV(value) as std,
           AVG(value) as mean
    FROM apm_results 
    WHERE season = '20252026'
    GROUP BY metric_name
    ORDER BY metric_name
""").fetchall()

print(f"{'Metric':<45} | Games | Players | Std      | Mean")
print("-" * 90)
for m, n, max_g, min_g, std, mean in result:
    std_str = f"{std:.6f}" if std is not None else "0.000000"
    mean_str = f"{mean:.6f}" if mean is not None else "0.000000"
    games_str = f"{max_g}" if max_g == min_g else f"{min_g}-{max_g}"
    print(f"{m:<45} | {games_str:>5} | {n:>7} | {std_str:>8} | {mean_str:>8}")

con.close()

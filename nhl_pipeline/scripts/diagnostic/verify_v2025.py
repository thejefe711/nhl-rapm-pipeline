"""
Verify all metrics for 2025-2026 have 459 games
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    print("Verifying 2025-26 RAPM Results (Full Suite):")
    print("-" * 60)
    
    results = con.execute("""
        SELECT metric_name, MIN(games_count), MAX(games_count), COUNT(*)
        FROM apm_results
        WHERE season = '20252026'
        GROUP BY metric_name
        ORDER BY metric_name
    """).fetchall()
    
    for m, min_g, max_g, count in results:
        status = "✅" if min_g == 459 else "❌"
        print(f"{status} {m:<45} | Games: {min_g}-{max_g} | Rows: {count}")

    con.close()

if __name__ == "__main__":
    main()

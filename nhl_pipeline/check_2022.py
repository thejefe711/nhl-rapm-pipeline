"""
Check if 20222023 RAPM data is complete in DuckDB
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    print("Checking 2022-23 RAPM results:")
    print("-" * 40)
    
    results = con.execute("""
        SELECT metric_name, games_count, COUNT(*)
        FROM apm_results
        WHERE season = '20222023'
        GROUP BY metric_name, games_count
        ORDER BY games_count DESC
        LIMIT 10
    """).fetchall()
    
    if not results:
        print("No results found for 2022-2023.")
    else:
        for metric, count, num_players in results:
            print(f"Metric: {metric:<40} | Games: {count:>5} | Players: {num_players:>5}")

    con.close()

if __name__ == "__main__":
    main()

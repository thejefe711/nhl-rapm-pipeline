"""
Check completion status for historical seasons in DuckDB
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    seasons = ['20252026', '20242025', '20232024', '20222023', '20212022', '20202021']
    
    print(f"{'Season':<10} | {'Metric Count':<15} | {'Avg Games':<10} | {'Status'}")
    print("-" * 60)
    
    for season in seasons:
        res = con.execute("""
            SELECT COUNT(DISTINCT metric_name), AVG(games_count)
            FROM apm_results
            WHERE season = ?
        """, [season]).fetchone()
        
        count, avg_games = res if res else (0, 0)
        
        status = "✅ COMPLETE" if avg_games and avg_games > 400 else "❌ INCOMPLETE"
        if season == '20252026' and avg_games > 400: status = "✅ FIXED"
        
        print(f"{season:<10} | {count:<15} | {avg_games:>10.1f} | {status}")

    con.close()

if __name__ == "__main__":
    main()

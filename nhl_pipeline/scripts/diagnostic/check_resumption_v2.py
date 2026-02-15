"""
Safe check for DuckDB contents
"""
import duckdb
import os

def main():
    try:
        con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
        
        seasons = ['20252026', '20242025', '20232024', '20222023', '20212022', '20202021']
        
        print(f"{'Season':<10} | {'Metrics':<8} | {'Avg Games':<10} | {'Status'}")
        print("-" * 55)
        
        for season in seasons:
            res = con.execute("SELECT COUNT(DISTINCT metric_name), AVG(games_count) FROM apm_results WHERE season = ?", [season]).fetchone()
            count = res[0] if res[0] is not None else 0
            avg_games = res[1] if res[1] is not None else 0
            
            status = "✅ FIXED" if avg_games > 400 else "❌ INCOMPLETE"
            if season == '20202021' and avg_games > 400: status = "✅ FIXED" # 20202021 has 868 games naturally
            
            print(f"{season:<10} | {count:<8} | {avg_games:>10.1f} | {status}")
            
        con.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

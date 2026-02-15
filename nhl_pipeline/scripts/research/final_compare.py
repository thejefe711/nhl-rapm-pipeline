"""
Standardized check of McDavid vs Jarvis
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    players = [
        {"id": 8478402, "name": "Connor McDavid"},
        {"id": 8482093, "name": "Seth Jarvis"}
    ]
    
    print(f"{'Player':<20} | {'Metric':<40} | {'Value':<10}")
    print("-" * 75)
    
    for p in players:
        results = con.execute("""
            SELECT metric_name, value 
            FROM apm_results 
            WHERE player_id = ? 
              AND season = '20252026' 
              AND (metric_name LIKE '%takeaway%' OR metric_name LIKE '%giveaway%')
        """, [p["id"]]).fetchall()
        
        for m, v in results:
            print(f"{p['name']:<20} | {m:<40} | {v:>10.4f}")

    con.close()

if __name__ == "__main__":
    main()

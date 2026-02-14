"""
Query giveaway/turnover RAPM for McDavid and Slavin
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    players = [
        {"id": 8478402, "name": "Connor McDavid"},
        {"id": 8476958, "name": "Jaccob Slavin"}
    ]
    
    metrics = [
        "giveaway_to_xg_swing_rapm_5v5_w10",
        "turnover_to_xg_swing_rapm_5v5_w10",
        "takeaway_to_xg_swing_rapm_5v5_w10"
    ]
    
    print(f"{'Player':<20} | {'Metric':<40} | {'Value':<10} | {'Games':<5}")
    print("-" * 85)
    
    for p in players:
        for metric in metrics:
            result = con.execute("""
                SELECT value, games_count
                FROM apm_results
                WHERE season = '20252026'
                  AND player_id = ?
                  AND metric_name = ?
            """, [p["id"], metric]).fetchone()
            
            if result:
                val, games = result
                print(f"{p['name']:<20} | {metric:<40} | {val:>10.4f} | {games:>5}")
            else:
                print(f"{p['name']:<20} | {metric:<40} | {'N/A':>10}")
        print("-" * 85)

    con.close()

if __name__ == "__main__":
    main()

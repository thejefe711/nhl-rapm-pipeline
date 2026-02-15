"""
Query giveaway/turnover RAPM for Seth Jarvis
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    player_id = 8482093
    name = "Seth Jarvis"
    
    metrics = [
        "giveaway_to_xg_swing_rapm_5v5_w10",
        "turnover_to_xg_swing_rapm_5v5_w10",
        "takeaway_to_xg_swing_rapm_5v5_w10"
    ]
    
    print(f"{'Player':<20} | {'Metric':<40} | {'Value':<10} | {'Games':<5}")
    print("-" * 85)
    
    for metric in metrics:
        result = con.execute("""
            SELECT value, games_count
            FROM apm_results
            WHERE season = '20252026'
              AND player_id = ?
              AND metric_name = ?
        """, [player_id, metric]).fetchone()
        
        if result:
            val, games = result
            print(f"{name:<20} | {metric:<40} | {val:>10.4f} | {games:>5}")
        else:
            print(f"{name:<20} | {metric:<40} | {'N/A':>10}")

    con.close()

if __name__ == "__main__":
    main()

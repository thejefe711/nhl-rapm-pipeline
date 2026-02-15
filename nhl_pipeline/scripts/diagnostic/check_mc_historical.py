"""
Check takeawaay for McDavid across all seasons
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    mcdavid_id = 8478402
    
    results = con.execute("""
        SELECT season, metric_name, value, games_count
        FROM apm_results
        WHERE player_id = ?
          AND metric_name = 'takeaway_to_xg_swing_rapm_5v5_w10'
        ORDER BY season DESC
    """, [mcdavid_id]).fetchall()
    
    for s, m, v, g in results:
        print(f"{s} | {m} | {v:.6f} | {g}")

    con.close()

if __name__ == "__main__":
    main()

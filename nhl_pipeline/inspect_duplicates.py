"""
Check for duplicates and inspect data consistency for McDavid
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    mcdavid_id = 8478402
    
    print("Entries for McDavid Takeaway metrics (2025-2026):")
    print("-" * 80)
    
    # Check for all variations and look for duplicates
    results = con.execute("""
        SELECT metric_name, value, games_count, COUNT(*) as occurs
        FROM apm_results
        WHERE season = '20252026'
          AND player_id = ?
          AND (metric_name LIKE '%takeaway%' OR metric_name LIKE '%giveaway%')
        GROUP BY metric_name, value, games_count
        ORDER BY metric_name
    """, [mcdavid_id]).fetchall()
    
    for m, v, g, o in results:
        print(f"[{o}x] {m:<45} | {v:>10.4f} | {g:>5} games")

    print("\nLeague Top 5 Takeaway values currently in DB:")
    top_5 = con.execute("""
        SELECT player_id, metric_name, value
        FROM apm_results
        WHERE season = '20252026'
          AND metric_name = 'takeaway_to_xg_swing_rapm_5v5_w10'
        ORDER BY value DESC
        LIMIT 5
    """).fetchall()
    for r in top_5:
        print(r)

    con.close()

if __name__ == "__main__":
    main()

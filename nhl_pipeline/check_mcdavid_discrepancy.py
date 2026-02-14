"""
Check all metrics for Connor McDavid to find leaderboard match
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    mcdavid_id = 8478402
    
    print(f"Metrics for Connor McDavid (2025-26):")
    print("=" * 60)
    
    results = con.execute("""
        SELECT metric_name, value, games_count
        FROM apm_results
        WHERE season = '20252026'
          AND player_id = ?
          AND metric_name LIKE '%takeaway%'
        ORDER BY value DESC
    """, [mcdavid_id]).fetchall()
    
    for m, v, g in results:
        print(f"{m:<45} | {v:>10.4f} | {g:>5}")

    con.close()

if __name__ == "__main__":
    main()

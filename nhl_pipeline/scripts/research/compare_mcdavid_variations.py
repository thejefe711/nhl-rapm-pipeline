"""
Compare takeaway variations for McDavid
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    mcdavid_id = 8478402
    
    print("McDavid Takeaway Metric Comparisons (2025-2026):")
    print("-" * 70)
    
    results = con.execute("""
        SELECT metric_name, value, games_count
        FROM apm_results
        WHERE player_id = ?
          AND season = '20252026'
          AND metric_name LIKE '%takeaway%'
        ORDER BY metric_name
    """, [mcdavid_id]).fetchall()
    
    for m, v, g in results:
        print(f"{m:<50} | {v:>10.4f} | {g:>5}")

    con.close()

if __name__ == "__main__":
    main()

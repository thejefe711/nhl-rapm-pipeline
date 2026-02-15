"""
Diagnose zero-variance - write to file with UTF-8 encoding
"""
import duckdb

con = duckdb.connect("nhl_canonical.duckdb", read_only=True)

with open("zero_variance_report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("COMPARING WORKING VS BROKEN xG SWING METRICS\n")
    f.write("=" * 70 + "\n")
    
    result = con.execute("""
        SELECT metric_name, 
               COUNT(*) as n,
               AVG(value) as mean,
               STDDEV(value) as std,
               MIN(value) as min_val,
               MAX(value) as max_val,
               MAX(games_count) as games
        FROM apm_results 
        WHERE season = '20252026'
          AND metric_name LIKE '%xg_swing%'
        GROUP BY metric_name
        ORDER BY std DESC
    """).fetchall()
    
    f.write(f"\n{'Metric':<50} | Games | Std      | Range\n")
    f.write("-" * 90 + "\n")
    for m, n, mean, std, min_v, max_v, games in result:
        range_str = f"{min_v:.4f} to {max_v:.4f}" if min_v is not None else "N/A"
        std_str = f"{std:.6f}" if std is not None else "0.000000"
        status = "BROKEN" if std is not None and std < 0.0001 else "OK"
        f.write(f"{m:<50} | {games:>5} | {std_str:>8} | {range_str} [{status}]\n")
    
    # Check if all values are identical
    f.write("\n" + "=" * 70 + "\n")
    f.write("SAMPLE VALUES FROM BROKEN METRICS\n")
    f.write("=" * 70 + "\n")
    
    for metric in ['giveaway_to_xg_swing_rapm_5v5_w10', 'turnover_to_xg_swing_rapm_5v5_w10']:
        result = con.execute(f"""
            SELECT player_id, value
            FROM apm_results 
            WHERE season = '20252026'
              AND metric_name = '{metric}'
            ORDER BY value DESC
            LIMIT 5
        """).fetchall()
        
        f.write(f"\n{metric} (top 5):\n")
        for pid, val in result:
            f.write(f"  Player {pid}: {val}\n")
        
        # Check if all same value
        unique = con.execute(f"""
            SELECT COUNT(DISTINCT value)
            FROM apm_results 
            WHERE season = '20252026'
              AND metric_name = '{metric}'
        """).fetchone()
        f.write(f"  Distinct values: {unique[0]}\n")

con.close()
print("Report written to zero_variance_report.txt")

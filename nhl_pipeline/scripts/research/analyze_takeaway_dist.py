"""
Analyze the distribution of Takeaway RAPM for 2025-26 using IDs
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    metric = "takeaway_to_xg_swing_rapm_5v5_w10"
    
    # Get stats
    stats = con.execute(f"""
        SELECT 
            MIN(value) as min_val,
            MAX(value) as max_val,
            AVG(value) as avg_val,
            STDDEV(value) as std_val,
            COUNT(*) as count
        FROM apm_results
        WHERE season = '20252026' AND metric_name = '{metric}'
    """).fetchone()
    
    min_v, max_v, avg_v, std_v, count = stats
    
    # Get McDavid and Slavin values
    mcdavid_id = 8478402
    slavin_id = 8476958
    
    mcdavid_val = con.execute(f"SELECT value FROM apm_results WHERE season = '20252026' AND metric_name = '{metric}' AND player_id = {mcdavid_id}").fetchone()
    slavin_val = con.execute(f"SELECT value FROM apm_results WHERE season = '20252026' AND metric_name = '{metric}' AND player_id = {slavin_id}").fetchone()
    
    print(f"League Stats for {metric}:")
    print(f"  Avg: {avg_v:.6f}")
    print(f"  Std: {std_v:.6f}")
    print(f"  Max: {max_v:.6f}")
    print(f"  Min: {min_v:.6f}")
    
    if mcdavid_val:
        m_val = mcdavid_val[0]
        m_z = (m_val - avg_v) / std_v
        print(f"\nConnor McDavid: {m_val:.6f} (Z-score: {m_z:.2f})")
        
    if slavin_val:
        s_val = slavin_val[0]
        s_z = (s_val - avg_v) / std_v
        print(f"Jaccob Slavin:  {s_val:.6f} (Z-score: {s_z:.2f})")
    
    # How many players have > 0.07?
    high_count = con.execute(f"SELECT COUNT(*) FROM apm_results WHERE season = '20252026' AND metric_name = '{metric}' AND value >= 0.07").fetchone()[0]
    print(f"\nPlayers with value >= 0.07: {high_count} ({high_count/count:.1%})")

    con.close()

if __name__ == "__main__":
    main()

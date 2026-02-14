"""
Final Comprehensive Verification for all seasons
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    seasons = con.execute("SELECT DISTINCT season FROM apm_results ORDER BY season DESC").fetchall()
    
    print(f"{'Season':<10} | {'Metrics':<8} | {'Min Games':<10} | {'Max Games':<10} | {'Avg Games':<10}")
    print("-" * 65)
    
    for (season,) in seasons:
        res = con.execute("""
            SELECT 
                COUNT(DISTINCT metric_name),
                MIN(games_count),
                MAX(games_count),
                AVG(games_count)
            FROM apm_results
            WHERE season = ?
        """, [season]).fetchone()
        
        count, min_g, max_g, avg_g = res
        print(f"{season:<10} | {count:<8} | {min_g:>10} | {max_g:>10} | {avg_g:>10.1f}")

    print("\nVariance Check (Giveaway xG Swing 5v5):")
    print("-" * 40)
    for (season,) in seasons:
        var_res = con.execute("""
            SELECT STDDEV(value)
            FROM apm_results
            WHERE season = ? AND metric_name = 'giveaway_to_xg_swing_rapm_5v5_w10'
        """, [season]).fetchone()
        std = var_res[0] if var_res[0] is not None else 0
        status = "✅ OK" if std > 0.01 else "❌ LOW VAR"
        print(f"{season}: StdDev = {std:.4f} | {status}")

    con.close()

if __name__ == "__main__":
    main()

import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

season = '20252026'

print(f"=== Data Quality Check for {season} ===")

# Check xg_off_rapm_5v5
xg_off_stats = con.execute(f"""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN value = 0 THEN 1 ELSE 0 END) as zero_values,
        AVG(value) as avg_value,
        MAX(value) as max_value,
        MIN(value) as min_value
    FROM apm_results 
    WHERE season = '{season}' AND metric_name = 'xg_off_rapm_5v5'
""").df()
print("\nxg_off_rapm_5v5 stats:")
print(xg_off_stats.to_string(index=False))

# Check corsi_off_rapm_5v5
corsi_off_stats = con.execute(f"""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN value = 0 THEN 1 ELSE 0 END) as zero_values,
        AVG(value) as avg_value,
        MAX(value) as max_value,
        MIN(value) as min_value
    FROM apm_results 
    WHERE season = '{season}' AND metric_name = 'corsi_off_rapm_5v5'
""").df()
print("\ncorsi_off_rapm_5v5 stats:")
print(corsi_off_stats.to_string(index=False))

# Check how many players have non-zero xG
non_zero_xg_players = con.execute(f"""
    SELECT COUNT(DISTINCT player_id) 
    FROM apm_results 
    WHERE season = '{season}' AND metric_name = 'xg_off_rapm_5v5' AND value != 0
""").fetchone()[0]
print(f"\nPlayers with non-zero xg_off in {season}: {non_zero_xg_players}")

con.close()

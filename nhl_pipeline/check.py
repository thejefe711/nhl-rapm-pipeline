import duckdb

con = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb', read_only=True)

print("=== 2025-26 Corsi RAPM distribution after latest re-run ===")
print(con.execute("""
    SELECT 
        COUNT(*) as n_players,
        round(MIN(value),2) as min,
        round(percentile_cont(0.9) WITHIN GROUP (ORDER BY value),2) as p90,
        round(MAX(value),2) as max,
        round(STDDEV(value),2) as std,
        COUNT(CASE WHEN value > 5 THEN 1 END) as players_above_5,
        COUNT(CASE WHEN value > 3 THEN 1 END) as players_above_3
    FROM apm_results
    WHERE metric_name = 'corsi_rapm_5v5' AND season = '20252026'
""").df().to_string())

print("\n=== Top 15 for 2025-26 with timestamps ===")
print(con.execute("""
    SELECT p.full_name, round(a.value,3) as value, a.toi_seconds, a.created_at
    FROM apm_results a
    LEFT JOIN players p ON a.player_id = p.player_id
    WHERE a.metric_name = 'corsi_rapm_5v5' AND a.season = '20252026'
    ORDER BY a.value DESC
    LIMIT 15
""").df().to_string())

print("\n=== created_at timestamps for 20252026 corsi_rapm_5v5 ===")
print(con.execute("""
    SELECT 
        MIN(created_at) as oldest,
        MAX(created_at) as newest,
        COUNT(DISTINCT created_at) as distinct_timestamps
    FROM apm_results
    WHERE metric_name = 'corsi_rapm_5v5' AND season = '20252026'
""").df().to_string())

print("\n=== Compare same season for xg_off_rapm_5v5 (also re-run today) ===")
print(con.execute("""
    SELECT round(MAX(value),2) as max, round(STDDEV(value),2) as std,
           MAX(created_at) as last_updated
    FROM apm_results
    WHERE metric_name = 'xg_off_rapm_5v5' AND season = '20252026'
""").df().to_string())

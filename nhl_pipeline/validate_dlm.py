import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

print("--- 1. Metrics and Players Processed ---")
q1 = "SELECT COUNT(DISTINCT metric_name) as metrics, COUNT(DISTINCT player_id) as players FROM dlm_rapm_estimates"
print(con.execute(q1).df())

print("\n--- 2. McDavid (8478402) 2024-2025 Signal vs Actual ---")
q2 = """
SELECT 
    metric_name,
    observed_value,
    filtered_mean,
    smoothed_mean,
    ROUND(filtered_var, 4) as uncertainty
FROM dlm_rapm_estimates
WHERE player_id = 8478402
AND season = '20242025'
AND metric_name IN ('xg_off_rapm_5v5', 'corsi_off_rapm_5v5', 'goals_off_rapm_5v5')
ORDER BY metric_name;
"""
print(con.execute(q2).df())

print("\n--- 3. Elite Defensemen Comparison (2024-2025) ---")
q3 = """
SELECT 
    p.full_name,
    d.observed_value,
    d.smoothed_mean,
    d.n_seasons
FROM dlm_rapm_estimates d
JOIN players p ON d.player_id = p.player_id
WHERE d.metric_name = 'xg_off_rapm_5v5'
AND d.season = '20242025'
AND p.full_name IN ('Adam Fox', 'Quinn Hughes', 'Cale Makar')
ORDER BY d.smoothed_mean DESC;
"""
print(con.execute(q3).df())

print("\n--- 4. Variance Reduction (Smoothing Effect) ---")
q4 = """
SELECT 
    metric_name,
    ROUND(STDDEV(observed_value), 4) as std_observed,
    ROUND(STDDEV(smoothed_mean), 4) as std_smoothed
FROM dlm_rapm_estimates
WHERE season = '20242025'
AND metric_name IN ('xg_off_rapm_5v5', 'corsi_off_rapm_5v5')
GROUP BY metric_name
ORDER BY metric_name;
"""
print(con.execute(q4).df())

con.close()

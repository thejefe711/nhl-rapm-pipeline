import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

print("=== TOI and Goals for Hughes, Makar, Fox (2024-2025) ===")
players = ['Quinn Hughes', 'Cale Makar', 'Adam Fox']
q_toi = """
SELECT 
    p.full_name,
    a.metric_name,
    a.value,
    a.toi_seconds / 60.0 as toi_minutes,
    a.games_count
FROM apm_results a
JOIN players p ON a.player_id = p.player_id
WHERE p.full_name IN ('Quinn Hughes', 'Cale Makar', 'Adam Fox')
AND a.season = '20242025'
AND a.metric_name IN ('corsi_off_rapm_5v5', 'xg_off_rapm_5v5')
ORDER BY p.full_name, a.metric_name;
"""
print(con.execute(q_toi).df().to_string(index=False))

print("\n=== xG Training Data Check (2025-2026) ===")
# Check if there are goals in 2025-2026 events
q_goals = """
SELECT 
    LEFT(CAST(game_id AS VARCHAR), 4) as season_prefix,
    event_type,
    COUNT(*) as count
FROM events
WHERE LEFT(CAST(game_id AS VARCHAR), 4) = '2025'
AND event_type IN ('SHOT', 'MISSED_SHOT', 'GOAL')
GROUP BY 1, 2;
"""
print(con.execute(q_goals).df().to_string(index=False))

con.close()

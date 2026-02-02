import duckdb
import pandas as pd

conn = duckdb.connect()
season = '20232024'
path = f"nhl_pipeline/staging/{season}/*_events.parquet"

players = {
    8478402: 'Connor McDavid',
    8476958: 'Jaccob Slavin',
    8476917: 'Adam Pelech'
}

results = []
for pid, name in players.items():
    query = f"""
    SELECT 
        '{name}' as player,
        COUNT(CASE WHEN player_1_id = {pid} THEN 1 END) as p1_count,
        COUNT(CASE WHEN player_2_id = {pid} THEN 1 END) as p2_count
    FROM read_parquet('{path}')
    WHERE event_type = 'BLOCKED_SHOT'
      AND CAST(game_id AS VARCHAR) LIKE '202302%'
    """
    res = conn.execute(query).df()
    results.append(res)

final_df = pd.concat(results)
with open('investigation_results.txt', 'w') as f:
    f.write("Comparison of player_1 (Shooter) vs player_2 (Blocker) for BLOCKED_SHOT:\n")
    f.write(final_df.to_string(index=False))
    f.write("\n\nChecking Takeaways and Giveaways (Regular Season):\n")
    for pid, name in players.items():
        query = f"""
        SELECT 
            '{name}' as player,
            event_type,
            COUNT(*) as count
        FROM read_parquet('{path}')
        WHERE event_type IN ('TAKEAWAY', 'GIVEAWAY')
          AND player_1_id = {pid}
          AND CAST(game_id AS VARCHAR) LIKE '202302%'
        GROUP BY 1, 2
        """
        res = conn.execute(query).df()
        f.write(res.to_string(index=False))
        f.write("\n")

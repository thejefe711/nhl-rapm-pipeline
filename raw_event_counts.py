import duckdb
import pandas as pd
from pathlib import Path

db_path = Path("nhl_pipeline/nhl_canonical.duckdb")
conn = duckdb.connect(str(db_path))

players = {
    8478402: "Connor McDavid",
    8476958: "Jaccob Slavin",
    8476917: "Adam Pelech"
}

event_types = ['BLOCKED_SHOT', 'TAKEAWAY', 'GIVEAWAY']
seasons = ['20232024', '20242025']

print("Raw Event Counts (Total for Season):")
print(f"{'Name':<15} | {'Season':<10} | {'Blocks':<6} | {'Takeaways':<9} | {'Giveaways':<9}")
print("-" * 60)

for pid, name in players.items():
    for season in seasons:
        # Query raw counts from events table
        query = f"""
        SELECT 
            event_type, 
            COUNT(*) as count
        FROM events e
        JOIN games g ON CAST(e.game_id AS VARCHAR) = CAST(g.game_id AS VARCHAR)
        WHERE CAST(e.player_1_id AS BIGINT) = {pid} 
        AND g.season = '{season}'
        AND e.event_type IN ('BLOCKED_SHOT', 'TAKEAWAY', 'GIVEAWAY')
        GROUP BY event_type
        """
        res = conn.execute(query).df()
        
        counts = {etype: 0 for etype in event_types}
        for _, row in res.iterrows():
            counts[row['event_type']] = row['count']
            
        print(f"{name:<15} | {season:<10} | {counts['BLOCKED_SHOT']:<6} | {counts['TAKEAWAY']:<9} | {counts['GIVEAWAY']:<9}")

conn.close()

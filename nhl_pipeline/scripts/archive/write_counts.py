import pandas as pd
import duckdb
import glob

ids = [8478402, 8476958, 8476917]
names = {8478402: 'Connor McDavid', 8476958: 'Jaccob Slavin', 8476917: 'Adam Pelech'}
results = []
conn = duckdb.connect()

for season in ['20232024', '20242025']:
    path = f"nhl_pipeline/staging/{season}/*_events.parquet"
    query = f"""
    SELECT 
        player_1_id, 
        event_type, 
        COUNT(*) as count
    FROM read_parquet('{path}')
    WHERE player_1_id IN (8478402, 8476958, 8476917)
      AND event_type IN ('BLOCKED_SHOT', 'TAKEAWAY', 'GIVEAWAY')
    GROUP BY 1, 2
    """
    df = conn.execute(query).df()
    df['season'] = season
    results.append(df)

final_df = pd.concat(results)
final_df['player_name'] = final_df['player_1_id'].map(names)
final_df = final_df[['season', 'player_name', 'event_type', 'count']].sort_values(['season', 'player_name', 'event_type'])
final_df.to_csv('final_counts.csv', index=False)
print(final_df.to_string(index=False))

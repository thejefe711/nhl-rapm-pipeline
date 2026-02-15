import duckdb
import pandas as pd

conn = duckdb.connect()
players = {
    8478402: 'Connor McDavid',
    8476958: 'Jaccob Slavin',
    8476917: 'Adam Pelech'
}

all_results = []
for season in ['20232024', '20242025']:
    path = f"nhl_pipeline/staging/{season}/*_events.parquet"
    
    # Corrected Blocks (player_2_id)
    query_blocks = f"""
    SELECT 
        player_2_id as player_id,
        'BLOCKED_SHOT' as event_type,
        COUNT(*) as count
    FROM read_parquet('{path}')
    WHERE event_type = 'BLOCKED_SHOT'
      AND player_2_id IN (8478402, 8476958, 8476917)
      AND CAST(game_id AS VARCHAR) LIKE '{season[:4]}02%'
    GROUP BY 1, 2
    """
    
    # Takeaways and Giveaways (player_1_id)
    query_others = f"""
    SELECT 
        player_1_id as player_id,
        event_type,
        COUNT(*) as count
    FROM read_parquet('{path}')
    WHERE event_type IN ('TAKEAWAY', 'GIVEAWAY')
      AND player_1_id IN (8478402, 8476958, 8476917)
      AND CAST(game_id AS VARCHAR) LIKE '{season[:4]}02%'
    GROUP BY 1, 2
    """
    
    df_blocks = conn.execute(query_blocks).df()
    df_others = conn.execute(query_others).df()
    
    df = pd.concat([df_blocks, df_others])
    df['season'] = season
    all_results.append(df)

final_df = pd.concat(all_results)
final_df['player_name'] = final_df['player_id'].map(players)
final_df = final_df[['season', 'player_name', 'event_type', 'count']].sort_values(['season', 'player_name', 'event_type'])

with open('corrected_counts.txt', 'w') as f:
    f.write(final_df.to_string(index=False))
print(final_df.to_string(index=False))

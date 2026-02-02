import duckdb
import pandas as pd
import glob
import os

def get_counts(season):
    dir_path = f"nhl_pipeline/staging/{season}"
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} does not exist.")
        return pd.DataFrame()
        
    path = f"{dir_path}/*_events.parquet"
    conn = duckdb.connect()
    
    # Find a sample file to check columns and types
    files = glob.glob(path)
    if not files:
        print(f"No parquet files found in {dir_path}")
        return pd.DataFrame()
        
    sample_file = files[0]
    try:
        schema = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{sample_file}')").df()
        
        # Check if player_1_id is string or int
        p1_type = schema[schema['column_name'] == 'player_1_id']['column_type'].values[0]
    except Exception as e:
        print(f"Error reading schema for {season}: {e}")
        return pd.DataFrame()

    # Query for the specific players and event types
    # McDavid: 8478402, Slavin: 8476958, Pelech: 8476917
    # Event types: BLOCKED_SHOT, TAKEAWAY, GIVEAWAY
    
    ids = [8478402, 8476958, 8476917]
    id_str = ",".join(map(str, ids))
    
    query = f"""
    SELECT 
        player_1_id, 
        event_type, 
        COUNT(*) as count
    FROM read_parquet('{path}')
    WHERE player_1_id IN ({id_str})
      AND event_type IN ('BLOCKED_SHOT', 'TAKEAWAY', 'GIVEAWAY')
    GROUP BY 1, 2
    """
    
    # If it's a string type, we need quotes
    if 'VARCHAR' in p1_type.upper() or 'STRING' in p1_type.upper():
        id_str_quoted = ",".join([f"'{i}'" for i in ids])
        query = f"""
        SELECT 
            player_1_id, 
            event_type, 
            COUNT(*) as count
        FROM read_parquet('{path}')
        WHERE player_1_id IN ({id_str_quoted})
          AND event_type IN ('BLOCKED_SHOT', 'TAKEAWAY', 'GIVEAWAY')
        GROUP BY 1, 2
        """

    try:
        res = conn.execute(query).df()
        if not res.empty:
            res['season'] = season
        return res
    except Exception as e:
        print(f"Error querying {season}: {e}")
        return pd.DataFrame()

all_results = []
for season in ['20232024', '20242025']:
    print(f"Processing {season}...")
    df = get_counts(season)
    if df is not None and not df.empty:
        all_results.append(df)

if all_results:
    final_df = pd.concat(all_results)
    # Map IDs to names
    names = {8478402: 'Connor McDavid', 8476958: 'Jaccob Slavin', 8476917: 'Adam Pelech',
             '8478402': 'Connor McDavid', '8476958': 'Jaccob Slavin', '8476917': 'Adam Pelech'}
    final_df['player_name'] = final_df['player_1_id'].map(names)
    print("\nRaw Event Counts:")
    pd.set_option('display.max_rows', None)
    print(final_df[['season', 'player_name', 'event_type', 'count']].sort_values(['season', 'player_name', 'event_type']).to_string(index=False))
else:
    print("\nNo results found across all seasons.")

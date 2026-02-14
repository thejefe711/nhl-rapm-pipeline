import duckdb
import pandas as pd
import argparse

def view_player_stats(player_name_filters=None):
    con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
    
    base_query = """
    SELECT 
        pi.player_name,
        ar.season,
        ar.metric_name,
        ar.value,
        pi.position
    FROM apm_results ar
    JOIN player_info pi ON ar.player_id = pi.player_id
    """
    
    query = base_query
    if player_name_filters:
        # OR conditions for multiple names
        conditions = [f"pi.player_name ILIKE '%{name}%'" for name in player_name_filters]
        query += " WHERE " + " OR ".join(conditions)
        
    query += " ORDER BY ar.season DESC, pi.player_name, ar.metric_name"
    
    df = con.execute(query).fetchdf()
    con.close()
    
    if df.empty:
        print("No stats found.")
        return

    # Group by player and print blocks
    players = df['player_name'].unique()
    metrics = sorted(df['metric_name'].unique())
    seasons = sorted(df['season'].unique())
    
    for player in players:
        pdf = df[df['player_name'] == player]
        pid = df[df['player_name'] == player]['player_id'].iloc[0] if 'player_id' in df.columns else '?'
        pos = pdf['position'].iloc[0]
        
        print(f"\n{'='*80}")
        print(f" {player} ({pos})")
        print(f"{'='*80}")
        
        # Header
        header = f"{'Metric':<40s}"
        player_seasons = sorted(pdf['season'].unique())
        for s in player_seasons:
            header += f" | {str(s)[-4:]}"
        print(header)
        print("-" * len(header))
        
        for m in metrics:
            mdf = pdf[pdf['metric_name'] == m]
            if mdf.empty:
                continue
                
            line = f"{m:<40s}"
            for s in player_seasons:
                val_rows = mdf[mdf['season'] == s]
                if val_rows.empty:
                    line += " |     "
                else:
                    line += f" | {val_rows.iloc[0]['value']:+5.2f}"
            print(line)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="Partial player names to filter (e.g. 'mcdavid' 'jarvis')")
    args = parser.parse_args()
    
    view_player_stats(args.names)

import duckdb
import pandas as pd
import argparse

def diagnose_shift_fragmentation(shifts_df: pd.DataFrame, game_id: int):
    """
    Check if shifts are fragmented (same player, multiple short shifts).
    """
    game_shifts = shifts_df[shifts_df['game_id'] == game_id].copy()
    game_shifts = game_shifts.sort_values(['player_id', 'start_seconds'])
    
    print(f"\n=== SHIFT FRAGMENTATION CHECK (Game {game_id}) ===")
    
    # Look for back-to-back shifts by same player
    game_shifts['prev_end'] = game_shifts.groupby('player_id')['end_seconds'].shift(1)
    game_shifts['gap'] = game_shifts['start_seconds'] - game_shifts['prev_end']
    
    # Shifts that start within 2 seconds of previous shift ending
    fragments = game_shifts[game_shifts['gap'].between(0, 2)]
    
    print(f"Total shifts: {len(game_shifts)}")
    print(f"Fragmented shifts (gap < 2s): {len(fragments)}")
    print(f"Fragmentation rate: {len(fragments) / len(game_shifts) * 100:.1f}%")
    
    if len(fragments) > 0:
        print(f"\nSample fragmented shifts:")
        sample_player = fragments['player_id'].iloc[0]
        player_shifts = game_shifts[game_shifts['player_id'] == sample_player].head(10)
        print(player_shifts[['player_id', 'start_seconds', 'end_seconds', 'duration_seconds', 'gap']])
    
    # Count unique boundary times
    boundaries = set(game_shifts['start_seconds'].tolist() + game_shifts['end_seconds'].tolist())
    print(f"\nUnique boundary times: {len(boundaries)}")
    print(f"  If this is >300, shifts are over-fragmented")

def main():
    parser = argparse.ArgumentParser(description="Diagnose shift fragmentation")
    parser.add_argument("--season", type=int, default=20242025, help="Season to analyze")
    parser.add_argument("--limit", type=int, default=3, help="Number of games to analyze")
    args = parser.parse_args()

    conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
    
    # Get game IDs for the season (filter by prefix)
    season_start = str(args.season)[:4]
    query = f"SELECT DISTINCT game_id FROM shifts WHERE CAST(game_id AS VARCHAR) LIKE '{season_start}%' LIMIT {args.limit}"
    print(f"DEBUG: Executing query: {query}")
    games = conn.execute(query).fetchdf()
    
    if games.empty:
        print(f"No games found for season {args.season}")
        return

    game_ids = games['game_id'].tolist()
    
    for gid in game_ids:
        shifts = conn.execute(f"SELECT * FROM shifts WHERE game_id = {gid}").fetchdf()
        diagnose_shift_fragmentation(shifts, gid)

if __name__ == "__main__":
    main()

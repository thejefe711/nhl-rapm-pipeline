import pandas as pd
from pathlib import Path

def inspect_shifts():
    # Find a shift file
    staging_dir = Path("nhl_pipeline/staging")
    shift_files = list(staging_dir.glob("*/*_shifts.parquet"))
    
    if not shift_files:
        print("No shift files found.")
        return

    # Pick the first one
    shift_file = shift_files[0]
    print(f"Inspecting: {shift_file}")
    
    df = pd.read_parquet(shift_file)
    game_id = df['game_id'].iloc[0]
    
    print(f"Shifts in game {game_id}: {len(df)}")
    print(f"Shift duration stats:")
    print(df['duration_seconds'].describe())
    
    print(f"\nSample shifts:")
    print(df[['player_id', 'start_seconds', 'end_seconds', 'duration_seconds']].head(20))

if __name__ == "__main__":
    inspect_shifts()

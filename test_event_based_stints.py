import pandas as pd
import numpy as np
import glob
import os

def generate_stints_from_events(event_on_ice_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate stints directly from event on-ice data.
    A stint = consecutive events with the same 10 skaters.
    """
    
    df = event_on_ice_df.copy()
    
    # Filter to 5v5 only
    if "is_5v5" in df.columns:
        df = df[df["is_5v5"] == True].copy()
    else:
        print("Warning: 'is_5v5' column not found, skipping 5v5 filter")

    df = df.sort_values(["period", "period_seconds"]).reset_index(drop=True)
    
    if df.empty:
        return pd.DataFrame()
    
    # Build lineup identifier for each event
    home_cols = ["home_skater_1", "home_skater_2", "home_skater_3", "home_skater_4", "home_skater_5"]
    away_cols = ["away_skater_1", "away_skater_2", "away_skater_3", "away_skater_4", "away_skater_5"]
    
    # Ensure columns exist
    missing_cols = [c for c in home_cols + away_cols if c not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return pd.DataFrame()

    def get_lineup(row):
        home = frozenset(int(x) for x in row[home_cols] if pd.notna(x))
        away = frozenset(int(x) for x in row[away_cols] if pd.notna(x))
        return (home, away)
    
    df["lineup"] = df.apply(get_lineup, axis=1)
    
    # Detect lineup changes
    df["lineup_changed"] = df["lineup"] != df["lineup"].shift(1)
    df["stint_id"] = df["lineup_changed"].cumsum()
    
    # Aggregate to stint level
    stints = df.groupby("stint_id").agg(
        period=("period", "first"),
        start_s=("period_seconds", "first"),
        end_s=("period_seconds", "last"),
        home_players=("lineup", lambda x: list(x.iloc[0][0])),
        away_players=("lineup", lambda x: list(x.iloc[0][1])),
        event_count=("event_id", "count"),
    ).reset_index(drop=True)
    
    # Calculate duration
    # Duration = time from first event to first event of NEXT stint
    stints["end_s_adjusted"] = stints["start_s"].shift(-1)
    stints.loc[stints["end_s_adjusted"].isna(), "end_s_adjusted"] = stints["end_s"]
    
    # Handle period boundaries
    stints["next_period"] = stints["period"].shift(-1)
    period_boundary = stints["period"] != stints["next_period"]
    stints.loc[period_boundary, "end_s_adjusted"] = 1200  # End of period (approximate, usually 1200)
    
    # For the very last stint, we might not know the true end time if no event occurred at 1200.
    # But let's assume it goes to the last event time or 1200? 
    # The user's logic uses 'end_s' (last event time) if 'end_s_adjusted' is NaN (last row).
    # But if it's the last row of a period, we should probably cap at 1200.
    # Let's stick to user's logic first but ensure last row of period gets 1200 if it's not set.
    
    # Actually, the user's logic:
    # stints.loc[stints["end_s_adjusted"].isna(), "end_s_adjusted"] = stints["end_s"]
    # This handles the very last row of the DF.
    # Then:
    # stints.loc[period_boundary, "end_s_adjusted"] = 1200
    # This handles the last stint of periods 1, 2, (and 3 if not OT).
    
    stints["duration_s"] = stints["end_s_adjusted"] - stints["start_s"]
    
    # Clean up
    stints = stints[stints["duration_s"] > 0].copy()
    
    return stints[["period", "start_s", "end_s_adjusted", "duration_s", "home_players", "away_players", "event_count"]]

# Find the file
pattern = "**/*2024020001_event_on_ice.parquet"
files = glob.glob(pattern, recursive=True)
if not files:
    files = glob.glob(f"nhl_pipeline/{pattern}", recursive=True)

if not files:
    print("Could not find event_on_ice file.")
    exit(1)

file_path = files[0]
print(f"Reading {file_path}...")
event_on_ice = pd.read_parquet(file_path)

print("Generating stints...")
stints = generate_stints_from_events(event_on_ice)

print(f"Stint count: {len(stints)}")
if not stints.empty:
    print(f"Duration - min: {stints['duration_s'].min()}, median: {stints['duration_s'].median()}, max: {stints['duration_s'].max()}")
    print(f"\nFirst 10 stints:")
    print(stints.head(10))
else:
    print("No stints generated.")

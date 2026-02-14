"""
Check columns in event_on_ice parquet files
"""
import pandas as pd
from pathlib import Path

# Check one event_on_ice file
canonical_dir = Path("canonical/20252026")
files = list(canonical_dir.glob("*_event_on_ice.parquet"))[:1]

if files:
    df = pd.read_parquet(files[0])
    print(f"Columns in {files[0].name} ({len(df)} rows):")
    print("-" * 50)
    for col in sorted(df.columns):
        print(f"  - {col}")
    
    # Check for xG swing columns
    target_cols = ["giveaway_xg_swing", "turnover_xg_swing", "takeaway_xg_swing", "block_xg_swing", "faceoff_xg_swing"]
    print("\n\nxG swing column check:")
    for col in target_cols:
        if col in df.columns:
            nonzero = (df[col] != 0).sum()
            print(f"  {col}: FOUND ({nonzero}/{len(df)} non-zero values)")
        else:
            print(f"  {col}: MISSING")
else:
    print("No event_on_ice files found")

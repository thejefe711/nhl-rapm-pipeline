#!/usr/bin/env python3
"""Check strength state distribution in canonical on-ice data."""

import pandas as pd
from pathlib import Path

canonical_dir = Path(__file__).parent / "canonical" / "20252026"

# Get a sample file
onice_files = list(canonical_dir.glob("*_onice.parquet"))
print(f"Found {len(onice_files)} onice files for 20252026")

if onice_files:
    # Load first file
    df = pd.read_parquet(onice_files[0])
    print(f"\nSample file: {onice_files[0].name}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nRows: {len(df)}")
    
    if "home_skater_count" in df.columns and "away_skater_count" in df.columns:
        print("\nStrength state distribution:")
        combo = df.groupby(["home_skater_count", "away_skater_count"]).size().reset_index(name="count")
        print(combo.sort_values("count", ascending=False).head(20).to_string(index=False))
        
        # PP situations (unequal skaters, 3-5 each)
        pp_mask = (
            (df["home_skater_count"] >= 3) & (df["home_skater_count"] <= 5) &
            (df["away_skater_count"] >= 3) & (df["away_skater_count"] <= 5) &
            (df["home_skater_count"] != df["away_skater_count"])
        )
        pp_events = df[pp_mask]
        print(f"\nPP events (unequal 3-5 skaters): {len(pp_events)}")
        print(f"5v5 events: {len(df[(df['home_skater_count'] == 5) & (df['away_skater_count'] == 5)])}")
    else:
        print("No skater count columns found!")

# Check all files for total
print("\n\nChecking ALL 20252026 canonical files:")
total_5v5 = 0
total_pp = 0
for f in onice_files:
    df = pd.read_parquet(f)
    if "home_skater_count" in df.columns:
        hc = df["home_skater_count"]
        ac = df["away_skater_count"]
        total_5v5 += len(df[(hc == 5) & (ac == 5)])
        pp_mask = ((hc >= 3) & (hc <= 5) & (ac >= 3) & (ac <= 5) & (hc != ac))
        total_pp += len(df[pp_mask])

print(f"Total 5v5 events: {total_5v5}")
print(f"Total PP events: {total_pp}")

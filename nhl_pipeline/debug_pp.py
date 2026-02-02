#!/usr/bin/env python3
"""Debug PP stint generation."""

import pandas as pd
from pathlib import Path

canonical_dir = Path(__file__).parent / "canonical" / "20252026"
staging_dir = Path(__file__).parent / "staging" / "20252026"

# Get first game
game_ids = [f.stem.replace("_event_on_ice", "") for f in canonical_dir.glob("*_event_on_ice.parquet")]
game_id = game_ids[0] if game_ids else None

print(f"Testing game: {game_id}")

if game_id:
    onice_path = canonical_dir / f"{game_id}_event_on_ice.parquet"
    shifts_path = staging_dir / f"{game_id}_shifts.parquet"
    events_path = staging_dir / f"{game_id}_events.parquet"
    
    print(f"onice exists: {onice_path.exists()}")
    print(f"shifts exists: {shifts_path.exists()}")
    print(f"events exists: {events_path.exists()}")
    
    if shifts_path.exists() and onice_path.exists():
        shifts = pd.read_parquet(shifts_path)
        onice = pd.read_parquet(onice_path)
        events = pd.read_parquet(events_path)
        
        print(f"\nShifts: {len(shifts)} rows")
        print(f"On-ice events: {len(onice)} rows")
        print(f"Events: {len(events)} rows")
        
        # Check for PP scenarios in on-ice data
        if "home_skater_count" in onice.columns:
            hc = onice["home_skater_count"]
            ac = onice["away_skater_count"]
            
            print(f"\nSkater count distribution:")
            print(onice.groupby(["home_skater_count", "away_skater_count"]).size().head(15))
            
            pp_mask = ((hc >= 3) & (hc <= 5) & (ac >= 3) & (ac <= 5) & (hc != ac))
            print(f"\nPP events (on-ice): {pp_mask.sum()}")
            
            # Sample PP events
            if pp_mask.sum() > 0:
                print("\nSample PP events:")
                print(onice[pp_mask][["event_id", "home_skater_count", "away_skater_count"]].head(5))
        
        # Check shifts for period data
        print(f"\nShift periods: {shifts['period'].unique().tolist()}")
        
        # Check if shifts have team info
        print(f"Teams in shifts: {shifts['team_id'].unique().tolist()}")

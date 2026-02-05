"""
Diagnostic script to trace where goal/assist data is being lost in the pipeline.
"""
import pandas as pd
from pathlib import Path
import json

def main():
    staging_dir = Path('staging')
    canonical_dir = Path('canonical')
    season = '20252026'
    
    print("=" * 70)
    print("DIAGNOSTIC: TRACING GOAL DATA LOSS IN RAPM PIPELINE")
    print("=" * 70)
    
    # Get first game
    events_files = list(staging_dir.glob(f'{season}/*_events.parquet'))
    if not events_files:
        print("No events files found!")
        return
    
    events_file = events_files[0]
    game_id = events_file.stem.replace('_events', '')
    print(f"\nTracing game: {game_id}")
    
    # Step 1: Raw events
    events = pd.read_parquet(events_file)
    goals = events[events['event_type'] == 'GOAL']
    print(f"\n1. RAW EVENTS FILE:")
    print(f"   Total events: {len(events)}")
    print(f"   Goals: {len(goals)}")
    print(f"   Goal event_ids: {list(goals['event_id'].values)}")
    
    # Step 2: On-ice data
    on_ice_path = canonical_dir / season / f'{game_id}_event_on_ice.parquet'
    if not on_ice_path.exists():
        print(f"\n2. ON-ICE FILE: NOT FOUND!")
        return
    
    on_ice = pd.read_parquet(on_ice_path)
    print(f"\n2. ON-ICE FILE:")
    print(f"   Total events: {len(on_ice)}")
    print(f"   Columns: {list(on_ice.columns)}")
    
    # Check which goals are in on-ice
    goal_ids = set(goals['event_id'].values)
    on_ice_ids = set(on_ice['event_id'].values)
    matching_ids = goal_ids.intersection(on_ice_ids)
    missing_ids = goal_ids - on_ice_ids
    
    print(f"   Goal IDs found in on-ice: {len(matching_ids)} of {len(goal_ids)}")
    if missing_ids:
        print(f"   MISSING goal IDs: {missing_ids}")
    
    # Step 3: 5v5 filter
    if matching_ids:
        on_ice_goals = on_ice[on_ice['event_id'].isin(goal_ids)]
        goals_5v5 = on_ice_goals[on_ice_goals['is_5v5'] == True]
        print(f"\n3. 5V5 FILTER:")
        print(f"   Goals in on-ice: {len(on_ice_goals)}")
        print(f"   Goals at 5v5: {len(goals_5v5)}")
        print(f"   Filtered out: {len(on_ice_goals) - len(goals_5v5)}")
    
    # Step 4: Check validation status
    validation_path = Path('data/on_ice_validation.json')
    if validation_path.exists():
        data = json.loads(validation_path.read_text())
        this_game = [r for r in data if str(r.get('game_id')) == game_id]
        if this_game:
            print(f"\n4. GATE 2 VALIDATION:")
            print(f"   Status: {'PASSED' if this_game[0].get('all_passed') else 'FAILED'}")
        else:
            print(f"\n4. GATE 2 VALIDATION: Game not in validation file")
    
    # Step 5: Count across all validated games
    print("\n" + "=" * 70)
    print("AGGREGATE CHECK: ALL VALIDATED GAMES")
    print("=" * 70)
    
    total_raw_goals = 0
    total_5v5_goals = 0
    games_checked = 0
    
    for ef in events_files[:100]:  # Check first 100 games
        gid = ef.stem.replace('_events', '')
        oip = canonical_dir / season / f'{gid}_event_on_ice.parquet'
        
        if not oip.exists():
            continue
        
        ev = pd.read_parquet(ef)
        oi = pd.read_parquet(oip)
        
        gls = ev[ev['event_type'] == 'GOAL']
        total_raw_goals += len(gls)
        
        oi_gls = oi[oi['event_id'].isin(gls['event_id'].values)]
        total_5v5_goals += len(oi_gls[oi_gls['is_5v5'] == True])
        games_checked += 1
    
    print(f"Games checked: {games_checked}")
    print(f"Total raw goals: {total_raw_goals}")
    print(f"Total 5v5 goals: {total_5v5_goals}")
    print(f"5v5 percentage: {100 * total_5v5_goals / max(total_raw_goals, 1):.1f}%")
    print(f"")
    print(f"Expected 5v5 goals for 459 games: ~{int(459 * total_5v5_goals / games_checked)}")
    print(f"Actual stored in DB: 169")

if __name__ == "__main__":
    main()

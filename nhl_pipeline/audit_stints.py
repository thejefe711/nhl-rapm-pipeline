import pandas as pd
from pathlib import Path

def audit_stints():
    root = Path('.')
    staging_dir = root / "staging"
    canonical_dir = root / "canonical"
    
    seasons = sorted([d.name for d in staging_dir.iterdir() if d.is_dir()])
    
    print("=== Stint and Lineup Audit ===")
    for season in seasons:
        game_files = list((staging_dir / season).glob("*_shifts.parquet"))
        print(f"\nSeason: {season}")
        print(f"  Games found: {len(game_files)}")
        
        # We'll sample 50 games to estimate the total stints
        sample_size = min(50, len(game_files))
        total_stints_est = 0
        unique_lineups = set()
        
        for i in range(sample_size):
            game_id = game_files[i].stem.replace("_shifts", "")
            on_ice_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
            if not on_ice_path.exists():
                continue
            
            # This is a rough estimate of stints per game
            # In compute_corsi_apm.py, stints are built from shift changes
            df_shifts = pd.read_parquet(game_files[i])
            # Count unique combinations of start/end seconds in period 1-3
            stints_in_game = len(df_shifts[df_shifts['period'].isin([1,2,3])][['period', 'start_seconds', 'end_seconds']].drop_duplicates())
            total_stints_est += stints_in_game
            
        avg_stints = total_stints_est / sample_size if sample_size > 0 else 0
        total_est = avg_stints * len(game_files)
        print(f"  Estimated total stints: {int(total_est):,}")
        print(f"  Avg stints per game: {avg_stints:.1f}")

if __name__ == "__main__":
    audit_stints()

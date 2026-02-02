import pandas as pd
import numpy as np

class DeepValidator:
    @staticmethod
    def validate_stints(stints_df: pd.DataFrame):
        print("\n=== STINT VALIDATION ===")
        
        # Count per game
        if 'game_id' in stints_df.columns:
            stints_per_game = stints_df.groupby('game_id').size()
            print(f"Stints per game: min={stints_per_game.min()}, median={stints_per_game.median():.0f}, max={stints_per_game.max()}")
            print(f"  Expected: 50-150")
            
            # Total duration vs expected game time
            game_durations = stints_df.groupby('game_id')['duration_s'].sum()
            print(f"\nTotal stint duration per game: min={game_durations.min():.0f}s, median={game_durations.median():.0f}s, max={game_durations.max():.0f}s")
            print(f"  Expected: ~2500-3200s (42-53 min of 5v5)")
        else:
            print("WARN: 'game_id' not in stints DataFrame, skipping per-game checks.")
        
        # Duration
        if 'duration_s' in stints_df.columns:
            print(f"\nStint duration (seconds): min={stints_df['duration_s'].min():.1f}, median={stints_df['duration_s'].median():.1f}, max={stints_df['duration_s'].max():.1f}")
            print(f"  Expected: 2-180")
        
        # Player counts (assuming columns like home_skater_1...6 exist or similar)
        # The RAPM script uses specific column names, let's try to infer or use what we know
        # Based on compute_corsi_apm.py, columns are home_skater_1..6 etc.
        
        home_cols = [c for c in stints_df.columns if 'home_skater' in c]
        away_cols = [c for c in stints_df.columns if 'away_skater' in c]
        
        if home_cols and away_cols:
            # Count non-nulls
            home_counts = stints_df[home_cols].notna().sum(axis=1)
            away_counts = stints_df[away_cols].notna().sum(axis=1)
            
            bad_home = (home_counts != 5).sum()
            bad_away = (away_counts != 5).sum()
            
            print(f"\nStints with !=5 home skaters: {bad_home} ({bad_home/len(stints_df)*100:.2f}%)")
            print(f"Stints with !=5 away skaters: {bad_away} ({bad_away/len(stints_df)*100:.2f}%)")
            print(f"  Expected: 0% (for 5v5 stints)")
        else:
            print("\nWARN: Could not identify player columns for count validation.")

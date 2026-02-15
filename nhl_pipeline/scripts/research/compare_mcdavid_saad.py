import pandas as pd

def compare_players(name1, name2, season=20242025):
    cat_df = pd.read_csv('profile_data/player_categories.csv')
    rapm_df = pd.read_csv('profile_data/player_rapm_full.csv')
    
    players = [name1, name2]
    
    print(f"=== Comparison: {name1} vs {name2} ({season}) ===")
    
    # Category Percentiles
    print("\n--- Category Percentiles (Signal) ---")
    cat_res = cat_df[(cat_df['season'] == season) & (cat_df['full_name'].isin(players))]
    for _, row in cat_res.iterrows():
        print(f"\n{row['full_name']}:")
        print(f"  OFFENSE:       {row['OFFENSE_signal_percentile']:.1f}%")
        print(f"  DEFENSE:       {row['DEFENSE_signal_percentile']:.1f}%")
        print(f"  TRANSITION:    {row['TRANSITION_signal_percentile']:.1f}%")
        print(f"  SPECIAL TEAMS: {row['SPECIAL_TEAMS_signal_percentile']:.1f}%")
        print(f"  DISCIPLINE:    {row['DISCIPLINE_signal_percentile']:.1f}%")
        print(f"  FINISHING:     {row['FINISHING_signal_percentile']:.1f}%")
        
    # Raw RAPM Metrics
    print("\n--- Key RAPM Metrics (Signal) ---")
    rapm_res = rapm_df[(rapm_df['season'] == season) & (rapm_df['full_name'].isin(players))]
    for _, row in rapm_res.iterrows():
        print(f"\n{row['full_name']}:")
        print(f"  Corsi Off 5v5: {row['corsi_off_rapm_5v5_signal']:.3f}")
        print(f"  xG Off 5v5:    {row['xg_off_rapm_5v5_signal']:.3f}")
        print(f"  Corsi Def 5v5: {row['corsi_def_rapm_5v5_signal']:.3f}")
        print(f"  Goals 5v5:     {row['goals_rapm_5v5_signal']:.3f}")
        print(f"  PP Corsi Off:  {row['corsi_pp_off_rapm_signal']:.3f}")

if __name__ == "__main__":
    compare_players('Connor McDavid', 'Brandon Saad')

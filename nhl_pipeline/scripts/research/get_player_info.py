import pandas as pd

def get_player_info():
    cats_df = pd.read_csv('profile_data/player_categories.csv')
    sim_df = pd.read_csv('profile_data/player_similarity.csv')
    
    target_names = ['Sebastian Aho', 'Seth Jarvis', 'Sidney Crosby', 'Connor McDavid', 'Jordan Staal']
    
    # Career stats (sum games and TOI across all seasons in categories)
    # Actually, player_categories has one row per player-season.
    # We need to aggregate for career stats.
    
    print("=== Player Career Stats & Similarity ===")
    
    for name in target_names:
        # Get all instances of this name
        p_cats = cats_df[cats_df['full_name'] == name]
        if p_cats.empty:
            print(f"\n{name}: Not found in categories.")
            continue
            
        # Group by player_id to handle name collisions (Aho)
        for pid in p_cats['player_id'].unique():
            p_season_data = p_cats[p_cats['player_id'] == pid]
            pos = p_season_data['position_group'].iloc[0]
            total_games = p_season_data['games_count'].sum()
            total_toi_sec = p_season_data['toi_total'].sum()
            total_toi_min = total_toi_sec / 60
            
            print(f"\n{name} ({pos}, ID: {pid}):")
            print(f"  Career Games: {total_games}")
            print(f"  Total TOI:   {total_toi_min:.1f} min")
            
            # Get career similarity (season='Career')
            p_sim = sim_df[(sim_df['player_id'] == pid) & (sim_df['season'] == 'Career')]
            if not p_sim.empty:
                row = p_sim.iloc[0]
                print("  Top 5 Career Similarities:")
                for i in range(1, 6):
                    sim_name = row.get(f'similar_{i}')
                    sim_score = row.get(f'similar_{i}_score')
                    if pd.notna(sim_name):
                        print(f"    {i}. {sim_name} ({sim_score:.3f})")
            else:
                print("  Career similarity not found.")

if __name__ == "__main__":
    get_player_info()

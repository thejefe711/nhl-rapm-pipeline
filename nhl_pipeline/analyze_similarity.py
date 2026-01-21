import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_similarities():
    df = pd.read_csv('profile_data/player_similarity.csv')
    
    # 2025-2026 Top 5
    print("=== 2025-2026 Top 5 Similarities ===")
    for name in ['Connor McDavid', 'Seth Jarvis']:
        player_data = df[(df['season'] == 20252026) & (df['full_name'] == name)]
        if not player_data.empty:
            row = player_data.iloc[0]
            print(f"\n{name}:")
            for i in range(1, 6):
                sim_name = row.get(f'similar_{i}')
                sim_score = row.get(f'similar_{i}_score')
                if pd.notna(sim_name):
                    print(f"  {i}. {sim_name} ({sim_score:.3f})")
        else:
            print(f"\n{name}: Not found in 2025-2026 data")

def compute_career_similarity(target_name):
    print(f"\n=== Career Similarity to {target_name} ===")
    df = pd.read_csv('profile_data/player_categories.csv')
    
    # Categories to use for similarity
    categories = ['OFFENSE', 'DEFENSE', 'TRANSITION', 'SPECIAL_TEAMS', 'DISCIPLINE', 'FINISHING']
    feature_cols = [f"{cat}_signal_score" for cat in categories]
    
    # Aggregate by player (average across all seasons)
    player_profiles = df.groupby(['player_id', 'full_name', 'position_group'])[feature_cols].mean().reset_index()
    
    # Get target profile
    target_profiles = player_profiles[player_profiles['full_name'] == target_name]
    if target_profiles.empty:
        print(f"Target player {target_name} not found.")
        return
    
    for _, target_profile in target_profiles.iterrows():
        target_vec = target_profile[feature_cols].fillna(0).values.reshape(1, -1)
        target_pos = target_profile['position_group']
        player_id = target_profile['player_id']
        
        print(f"\nResults for {target_name} (ID: {player_id}, Pos: {target_pos}):")
        
        # Compare only within same position group
        peers = player_profiles[player_profiles['position_group'] == target_pos].copy()
        X = peers[feature_cols].fillna(0).values
        
        # Compute similarity
        sims = cosine_similarity(target_vec, X).flatten()
        peers['similarity'] = sims
        
        # Sort and exclude self
        results = peers[peers['player_id'] != player_id].sort_values('similarity', ascending=False).head(5)
        
        for i, (_, row) in enumerate(results.iterrows(), 1):
            print(f"  {i}. {row['full_name']} ({row['similarity']:.3f})")

if __name__ == "__main__":
    get_similarities()
    compute_career_similarity('Connor McDavid')
    compute_career_similarity('Logan Stankoven')
    compute_career_similarity('Sebastian Aho')

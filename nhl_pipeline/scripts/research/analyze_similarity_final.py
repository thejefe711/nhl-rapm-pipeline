import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_career_similarity(target_name, output_file):
    df = pd.read_csv('profile_data/player_categories.csv')
    categories = ['OFFENSE', 'DEFENSE', 'TRANSITION', 'SPECIAL_TEAMS', 'DISCIPLINE', 'FINISHING']
    feature_cols = [f"{cat}_signal_score" for cat in categories]
    player_profiles = df.groupby(['player_id', 'full_name', 'position_group'])[feature_cols].mean().reset_index()
    
    target_profiles = player_profiles[player_profiles['full_name'] == target_name]
    
    with open(output_file, 'a', encoding='utf-8') as f:
        if target_profiles.empty:
            f.write(f"Target player {target_name} not found.\n")
            return
        
        for _, target_profile in target_profiles.iterrows():
            target_vec = target_profile[feature_cols].fillna(0).values.reshape(1, -1)
            target_pos = target_profile['position_group']
            player_id = target_profile['player_id']
            
            f.write(f"\nCareer Similarity for {target_name} (ID: {player_id}, Pos: {target_pos}):\n")
            peers = player_profiles[player_profiles['position_group'] == target_pos].copy()
            X = peers[feature_cols].fillna(0).values
            sims = cosine_similarity(target_vec, X).flatten()
            peers['similarity'] = sims
            results = peers[peers['player_id'] != player_id].sort_values('similarity', ascending=False).head(5)
            
            for i, (_, row) in enumerate(results.iterrows(), 1):
                f.write(f"  {i}. {row['full_name']} ({row['similarity']:.3f})\n")

if __name__ == "__main__":
    output_file = 'similarity_results_final.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Career Similarity Results ===\n")
    compute_career_similarity('Logan Stankoven', output_file)
    compute_career_similarity('Sebastian Aho', output_file)

import pandas as pd
import numpy as np

def verify_similarities():
    df = pd.read_csv('profile_data/player_similarity.csv')
    
    # Check specific players for 2025-2026
    players = ['Connor McDavid', 'Jack Hughes', 'Sebastian Aho', 'Brandon Saad']
    season = 20252026
    
    print(f"=== Similarity Verification ({season}) ===")
    
    for name in players:
        player_data = df[(df['season'] == season) & (df['full_name'] == name)]
        if not player_data.empty:
            row = player_data.iloc[0]
            is_qual = row.get('is_qualified', 'N/A')
            print(f"\n{name} (Qualified: {is_qual}):")
            for i in range(1, 6):
                sim_name = row.get(f'similar_{i}')
                sim_score = row.get(f'similar_{i}_score')
                if pd.notna(sim_name):
                    print(f"  {i}. {sim_name} ({sim_score:.3f})")
        else:
            print(f"\n{name}: Not found in {season} data")

if __name__ == "__main__":
    verify_similarities()

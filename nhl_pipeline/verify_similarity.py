import pandas as pd
import numpy as np

def verify_similarities():
    df = pd.read_csv('profile_data/player_similarity.csv')
    
    # Check specific players for current season
    from profile_pipeline.config import CURRENT_SEASON
    season = CURRENT_SEASON
    
    print(f"=== Similarity Verification ({season}) ===")
    
    # Gold Standard expectations
    gold_standard = {
        "Connor McDavid": ["Leon Draisaitl", "Nathan MacKinnon", "David Pastrnak", "Auston Matthews"],
        "Jack Hughes": ["Nathan MacKinnon", "David Pastrnak", "Auston Matthews", "Leon Draisaitl"],
        "Sebastian Aho": ["Brayden Point", "David Pastrnak", "Sidney Crosby", "Aleksander Barkov"],
    }
    
    pass_count = 0
    fail_count = 0
    
    for name, expected in gold_standard.items():
        player_data = df[(df['season'] == season) & (df['full_name'] == name)]
        if not player_data.empty:
            row = player_data.iloc[0]
            is_qual = row.get('is_qualified', 'N/A')
            print(f"\n{name} (Qualified: {is_qual}):")
            
            matches = []
            for i in range(1, 6):
                sim_name = row.get(f'similar_{i}')
                sim_score = row.get(f'similar_{i}_score')
                if pd.notna(sim_name):
                    matches.append(sim_name)
                    print(f"  {i}. {sim_name} ({sim_score:.3f})")
            
            # Check if any expected player is in top 5
            found = [e for e in expected if e in matches]
            if found:
                print(f"  ✓ Found expected: {found}")
                pass_count += 1
            else:
                print(f"  ✗ MISSING expected: {expected}")
                fail_count += 1
        else:
            print(f"\n{name}: Not found in {season} data")
            fail_count += 1
    
    # Negative Test: Logan Stankoven should NOT be in Jack Hughes' matches
    print("\n=== Negative Test: Logan Stankoven ===")
    hughes_data = df[(df['season'] == season) & (df['full_name'] == 'Jack Hughes')]
    if not hughes_data.empty:
        row = hughes_data.iloc[0]
        matches = [row.get(f'similar_{i}') for i in range(1, 6) if pd.notna(row.get(f'similar_{i}'))]
        
        if "Logan Stankoven" in matches:
            print(f"  ✗ FAIL: Logan Stankoven IS in Jack Hughes' top 5 matches!")
            fail_count += 1
        else:
            print(f"  ✓ PASS: Logan Stankoven is NOT in Jack Hughes' top 5 matches.")
            pass_count += 1
    
    # Logan Stankoven's own qualification status
    stankoven = df[(df['season'] == season) & (df['full_name'] == 'Logan Stankoven')]
    if not stankoven.empty:
        is_qual = stankoven.iloc[0].get('is_qualified', False)
        print(f"  Logan Stankoven is_qualified: {is_qual}")
        if not is_qual:
            print(f"  ✓ Correctly flagged as NOT qualified.")
    else:
        print(f"  Logan Stankoven not found in {season} data.")
    
    print(f"\n=== Summary: {pass_count} passed, {fail_count} failed ===")

if __name__ == "__main__":
    verify_similarities()


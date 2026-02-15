import pandas as pd

def check_scores():
    df = pd.read_csv('profile_data/player_categories.csv')
    players = ['Connor McDavid', 'Nathan MacKinnon', 'Auston Matthews', 'Leon Draisaitl', 'Seth Jarvis', 'Jason Zucker', 'Brandon Saad']
    season = 20252026
    
    results = df[(df['season'] == season) & (df['full_name'].isin(players))]
    
    print(f"=== Category Scores (Signal, {season}) ===")
    for _, row in results.iterrows():
        print(f"\n{row['full_name']}:")
        print(f"  OFFENSE:    {row['OFFENSE_signal_score']:.3f}")
        print(f"  DEFENSE:    {row['DEFENSE_signal_score']:.3f}")
        print(f"  USAGE:      {row['USAGE_signal_score']:.3f}")
        print(f"  TRANSITION: {row['TRANSITION_signal_score']:.3f}")
        print(f"  SPECIAL:    {row['SPECIAL_TEAMS_signal_score']:.3f}")
        print(f"  DISCIPLINE: {row['DISCIPLINE_signal_score']:.3f}")
        print(f"  FINISHING:  {row['FINISHING_signal_score']:.3f}")

if __name__ == "__main__":
    check_scores()

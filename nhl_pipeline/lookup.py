import pandas as pd
df = pd.read_csv('profile_data/player_narratives.csv')
df['season'] = df['season'].astype(int)

for name in ['Quinn Hughes', 'Cale Makar']:
    player = df[(df['full_name'] == name) & (df['season'] == 20242025)]
    if player.empty:
        player = df[df['full_name'].str.contains(name.split()[1], case=False)]
        if not player.empty:
            print(f"{name} found in seasons: {player['season'].unique().tolist()}")
            player = player[player['season'] == player['season'].max()]
        else:
            print(f"{name}: NOT FOUND")
            continue
    
    row = player.iloc[0]
    print(f"\n{row['full_name']}:")
    print(f"  Offense: {row.get('OFFENSE_percentile', 0):.1f}%")
    print(f"  Defense: {row.get('DEFENSE_percentile', 0):.1f}%")
    print(f"  Transition: {row.get('TRANSITION_percentile', 0):.1f}%")
    print(f"  Special Teams: {row.get('SPECIAL_TEAMS_percentile', 0):.1f}%")
    print(f"  Discipline: {row.get('DISCIPLINE_percentile', 0):.1f}%")
    print(f"  Finishing: {row.get('FINISHING_percentile', 0):.1f}%")

import pandas as pd

df = pd.read_csv('profile_data/player_narratives.csv')
df['season'] = df['season'].astype(int)

# Find Chatfield
chatfield = df[(df['full_name'].str.contains('Chatfield', case=False)) & (df['season'] == 20242025)]

if chatfield.empty:
    chatfield = df[df['full_name'].str.contains('Chatfield', case=False)]
    if chatfield.empty:
        print('Chatfield not found')
    else:
        print(f"Found in seasons: {chatfield['season'].unique().tolist()}")
        chatfield = chatfield[chatfield['season'] == chatfield['season'].max()]

if not chatfield.empty:
    row = chatfield.iloc[0]
    print(f"Name: {row['full_name']}")
    print(f"Position: {row['position_group']}")
    print(f"Archetype: {row.get('archetype', 'Unknown')}")
    print(f"Percentiles:")
    print(f"  Offense: {row.get('OFFENSE_percentile', 0):.1f}%")
    print(f"  Defense: {row.get('DEFENSE_percentile', 0):.1f}%")
    print(f"  Transition: {row.get('TRANSITION_percentile', 0):.1f}%")
    print(f"  Special Teams: {row.get('SPECIAL_TEAMS_percentile', 0):.1f}%")
    print(f"  Discipline: {row.get('DISCIPLINE_percentile', 0):.1f}%")
    print(f"  Finishing: {row.get('FINISHING_percentile', 0):.1f}%")
    print(f"Similar: {row.get('similar_1')}, {row.get('similar_2')}, {row.get('similar_3')}")
    print(f"Narrative: {row.get('narrative', '')}")

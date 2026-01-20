import pandas as pd
import json

# Load data
df = pd.read_csv('profile_data/player_narratives.csv')
df['season'] = df['season'].astype(int)

# Find Seth Jarvis
jarvis = df[(df['full_name'] == 'Seth Jarvis') & (df['season'] == 20242025)]

if jarvis.empty:
    jarvis_all = df[df['full_name'].str.contains('Jarvis', case=False)]
    print(f"No exact match for 2024-2025. Found: {jarvis_all['full_name'].unique().tolist()}")
    if not jarvis_all.empty:
        jarvis = jarvis_all[jarvis_all['season'] == jarvis_all['season'].max()]

if not jarvis.empty:
    row = jarvis.iloc[0]
    
    profile = {
        'player_id': int(row['player_id']),
        'full_name': row['full_name'],
        'position': row['position_group'],
        'archetype': row.get('archetype', 'Unknown'),
        'percentiles': {
            'offense': round(row.get('OFFENSE_percentile', 0), 1),
            'defense': round(row.get('DEFENSE_percentile', 0), 1),
            'transition': round(row.get('TRANSITION_percentile', 0), 1),
            'special_teams': round(row.get('SPECIAL_TEAMS_percentile', 0), 1),
            'discipline': round(row.get('DISCIPLINE_percentile', 0), 1),
            'finishing': round(row.get('FINISHING_percentile', 0), 1),
        },
        'similar_players': [row.get('similar_1'), row.get('similar_2'), row.get('similar_3')],
        'regression_flag': bool(row.get('regression_flag', False)),
        'breakout_flag': bool(row.get('breakout_flag', False)),
        'narrative': row.get('narrative', '')
    }
    
    print(json.dumps(profile, indent=2, default=str))
else:
    print("Seth Jarvis not found in data")

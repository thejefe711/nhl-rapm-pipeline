import pandas as pd
df = pd.read_csv('profile_data/player_rapm_full.csv')
for name in ['Quinn Hughes', 'Cale Makar']:
    p = df[df['full_name'] == name].sort_values('season')
    print(f"\n=== {name} ===")
    print(p[['season', 'corsi_off_rapm_5v5', 'xg_off_rapm_5v5', 'corsi_def_rapm_5v5', 'xg_def_rapm_5v5']].to_string(index=False))

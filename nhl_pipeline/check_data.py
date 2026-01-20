import pandas as pd

df = pd.read_csv('profile_data/player_rapm_full.csv')
print(f'Rows: {len(df)}')
print(f'Position groups: {df["position_group"].value_counts().to_dict()}')
print(f'Seasons: {df["season"].unique().tolist()}')

# Check McDavid
mcdavid = df[df["full_name"] == "Connor McDavid"]
print(f'\nMcDavid rows: {len(mcdavid)}')
if not mcdavid.empty:
    print(mcdavid[["season", "position", "position_group", "corsi_off_rapm_5v5"]].to_string())

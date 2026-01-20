#!/usr/bin/env python3
"""Extract SAE latent report with top/bottom 10 for all 12 dimensions."""

import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

# Get player names
names = con.execute('SELECT player_id, full_name FROM players').df()
name_map = {int(r['player_id']): r['full_name'] for _, r in names.iterrows()}

# Get model info
mdf = con.execute("SELECT model_name, n_components FROM latent_models WHERE model_name = 'sae_apm_v0_k12_a1'").df()
n_components = int(mdf.iloc[0]['n_components'])

# Get latent skills for 2024-2025 season
ldf = con.execute("""
    SELECT player_id, dim_idx, value
    FROM latent_skills
    WHERE model_name = 'sae_apm_v0_k12_a1' AND season = '20242025'
""").df()

# Get dimension metadata
dim_meta = con.execute("""
    SELECT dim_idx, label, top_features_json
    FROM latent_dim_meta  
    WHERE model_name = 'sae_apm_v0_k12_a1'
""").df()
dim_labels = {int(r['dim_idx']): r['label'] for _, r in dim_meta.iterrows()}

ldf['player_id'] = ldf['player_id'].astype(int)
Z = ldf.pivot_table(index='player_id', columns='dim_idx', values='value', aggfunc='mean')

print('=' * 80)
print('SAE LATENT REPORT - 12 Dimensions - Season 2024-2025')
print('=' * 80)

for k in range(n_components):
    label = dim_labels.get(k, "Unknown")
    print(f'\n--- Dim {k}: {label} ---')
    vals = Z[k].dropna().sort_values(ascending=False)
    
    print('TOP 10:')
    for pid, v in vals.head(10).items():
        name = name_map.get(int(pid), str(pid))
        print(f'  {name}: {v:+.3f}')
    
    print('BOTTOM 10:')
    for pid, v in vals.tail(10).sort_values().items():
        name = name_map.get(int(pid), str(pid))
        print(f'  {name}: {v:+.3f}')

con.close()

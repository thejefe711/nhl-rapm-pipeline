#!/usr/bin/env python3
"""Simple SAE analysis - outputs to structured JSON file."""

import duckdb
import json
import numpy as np
from pathlib import Path

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

result = {}

# Model info
model = con.execute("SELECT * FROM latent_models WHERE model_name = 'sae_apm_v0_k12_a1'").df()
result['model'] = {
    'n_samples': int(model.iloc[0]['n_samples']),
    'n_components': int(model.iloc[0]['n_components']),
    'alpha': float(model.iloc[0]['alpha']),
    'features': json.loads(model.iloc[0]['features_json'])
}

# Dictionary coefficients
dictionary = np.array(json.loads(model.iloc[0]['dictionary_json']))
result['dictionary_shape'] = list(dictionary.shape)

# Dictionary sparsity per dim
dict_sparsity = []
for k in range(dictionary.shape[0]):
    row = dictionary[k]
    nonzero = int(np.sum(np.abs(row) > 0.01))
    max_coef = float(np.max(np.abs(row)))
    dict_sparsity.append({'dim': k, 'nonzero_features': nonzero, 'max_abs_coef': round(max_coef, 3)})
result['dictionary_sparsity'] = dict_sparsity

# Per-dimension latent stats
dim_stats = con.execute("""
    SELECT dim_idx, 
           COUNT(*) as n,
           ROUND(AVG(value), 4) as mean,
           ROUND(STDDEV(value), 4) as std,
           ROUND(AVG(ABS(value)), 4) as mean_abs,
           ROUND(SUM(CASE WHEN ABS(value) < 0.001 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_zero
    FROM latent_skills
    WHERE model_name = 'sae_apm_v0_k12_a1'
    GROUP BY dim_idx
    ORDER BY dim_idx
""").df()
result['latent_stats'] = dim_stats.to_dict(orient='records')

# Labels
labels = con.execute("SELECT dim_idx, label, stable_seasons, top_features_json FROM latent_dim_meta WHERE model_name = 'sae_apm_v0_k12_a1' ORDER BY dim_idx").df()
result['labels'] = labels.to_dict(orient='records')

# Inter-dimension correlations
wide = con.execute("""
    SELECT season, player_id, dim_idx, value
    FROM latent_skills
    WHERE model_name = 'sae_apm_v0_k12_a1'
""").df()
pivot = wide.pivot_table(index=['season', 'player_id'], columns='dim_idx', values='value')
corr = pivot.corr()
result['inter_dim_corr'] = corr.round(3).to_dict()

# Reconstruction analysis
features = result['model']['features']
scaler_mean = np.array(json.loads(model.iloc[0]['scaler_mean_json']))
scaler_scale = np.array(json.loads(model.iloc[0]['scaler_scale_json']))
result['scaler'] = {
    'mean': [round(x, 4) for x in scaler_mean],
    'scale': [round(x, 4) for x in scaler_scale]
}

# Get original data
placeholders = ", ".join(["?"] * len(features))
long_df = con.execute(f"""
    SELECT season, player_id, metric_name, value
    FROM apm_results
    WHERE metric_name IN ({placeholders})
""", features).df()

long_df["player_id"] = long_df["player_id"].astype(int)
X_df = long_df.pivot_table(index=["season", "player_id"], columns="metric_name", values="value", aggfunc="mean").reset_index()

for f in features:
    if f not in X_df.columns:
        X_df[f] = np.nan
X_df = X_df.dropna(subset=features).copy()

X = X_df[features].astype(float).values
Xs = (X - scaler_mean) / scaler_scale

# Align
X_df['key'] = X_df['season'].astype(str) + '_' + X_df['player_id'].astype(str)
wide['key'] = wide['season'].astype(str) + '_' + wide['player_id'].astype(str)
pivot_aligned = wide.pivot_table(index='key', columns='dim_idx', values='value')
X_df_indexed = X_df.set_index('key')
common_keys = pivot_aligned.index.intersection(X_df_indexed.index)

Z = pivot_aligned.loc[common_keys].values
X_aligned = X_df_indexed.loc[common_keys][features].values
Xs_aligned = (X_aligned - scaler_mean) / scaler_scale

# Reconstruct
X_reconstructed = Z @ dictionary

# Compute reconstruction error
mse = float(np.mean((Xs_aligned - X_reconstructed)**2))
ss_tot = float(np.sum((Xs_aligned - Xs_aligned.mean(axis=0))**2))
ss_res = float(np.sum((Xs_aligned - X_reconstructed)**2))
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

result['reconstruction'] = {
    'mse': round(mse, 4),
    'r2': round(r2, 4),
    'n_samples_aligned': int(len(common_keys))
}

# Per-feature R2
per_feature_r2 = {}
for i, f in enumerate(features):
    ss_res_f = float(np.sum((Xs_aligned[:, i] - X_reconstructed[:, i])**2))
    ss_tot_f = float(np.sum((Xs_aligned[:, i] - Xs_aligned[:, i].mean())**2))
    r2_f = 1 - ss_res_f / ss_tot_f if ss_tot_f > 0 else 0
    per_feature_r2[f] = round(r2_f, 3)
result['per_feature_r2'] = per_feature_r2

con.close()

# Write JSON
Path('reports/sae_analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print("Written to reports/sae_analysis.json")

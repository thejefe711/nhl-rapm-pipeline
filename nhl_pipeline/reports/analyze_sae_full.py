#!/usr/bin/env python3
"""Full SAE model analysis for review."""

import duckdb
import json
import numpy as np
import pandas as pd
from pathlib import Path

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
out_lines = []

def log(s=""):
    out_lines.append(str(s))
    print(s)

# Model info
model = con.execute("SELECT * FROM latent_models WHERE model_name = 'sae_apm_v0_k12_a1'").df()
log('=== MODEL INFO ===')
log(f"n_samples: {model.iloc[0]['n_samples']}")
log(f"n_components: {model.iloc[0]['n_components']}")
log(f"alpha: {model.iloc[0]['alpha']}")

features = json.loads(model.iloc[0]['features_json'])
log(f"features ({len(features)}):")
for f in features:
    log(f"  - {f}")

# Dictionary (coefficients) - shape (n_components, n_features)
dictionary = np.array(json.loads(model.iloc[0]['dictionary_json']))
log(f'\n=== DICTIONARY COEFFICIENTS ===')
log(f'Shape: {dictionary.shape} (n_dims x n_features)')

log('\nDictionary matrix (rows=dims, cols=features):')
log(f"Features: {features}")
for k in range(dictionary.shape[0]):
    row = dictionary[k]
    row_str = ", ".join([f"{v:+.3f}" for v in row])
    log(f"  Dim {k:2d}: [{row_str}]")

# Analyze sparsity of dictionary
log(f'\n=== DICTIONARY SPARSITY ===')
for k in range(dictionary.shape[0]):
    row = dictionary[k]
    nonzero = np.sum(np.abs(row) > 0.01)
    max_coef = np.max(np.abs(row))
    log(f'  Dim {k:2d}: {nonzero}/{len(row)} features active (|coef|>0.01), max|coef|={max_coef:.3f}')

# Overall stats per dimension
log(f'\n=== LATENT CODE STATS (pooled across seasons) ===')
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
log(dim_stats.to_string(index=False))

# Dimension labels from metadata
log(f'\n=== DIMENSION LABELS (auto-generated) ===')
labels = con.execute("SELECT dim_idx, label, stable_seasons, top_features_json FROM latent_dim_meta WHERE model_name = 'sae_apm_v0_k12_a1' ORDER BY dim_idx").df()
for _, row in labels.iterrows():
    log(f"  Dim {row['dim_idx']:2d}: {row['label']:25s} (stable in {row['stable_seasons']} seasons) | top: {row['top_features_json']}")

# Correlation matrix between dimensions
log(f'\n=== INTER-DIMENSION CORRELATIONS ===')
wide = con.execute("""
    SELECT season, player_id, dim_idx, value
    FROM latent_skills
    WHERE model_name = 'sae_apm_v0_k12_a1'
""").df()
pivot = wide.pivot_table(index=['season', 'player_id'], columns='dim_idx', values='value')
corr = pivot.corr()
log("(Should be near-diagonal. High off-diagonal = dimensions are redundant)")
log(corr.round(2).to_string())

# Check reconstruction error
log(f'\n=== RECONSTRUCTION ANALYSIS ===')
scaler_mean = np.array(json.loads(model.iloc[0]['scaler_mean_json']))
scaler_scale = np.array(json.loads(model.iloc[0]['scaler_scale_json']))
log(f"Scaler mean: {[round(x, 4) for x in scaler_mean]}")
log(f"Scaler scale: {[round(x, 4) for x in scaler_scale]}")

# Get original data inline
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

# Get latent codes for same player-seasons
X_df['key'] = X_df['season'].astype(str) + '_' + X_df['player_id'].astype(str)
wide['key'] = wide['season'].astype(str) + '_' + wide['player_id'].astype(str)

# Align latent codes with features
pivot_aligned = wide.pivot_table(index='key', columns='dim_idx', values='value')
X_df_indexed = X_df.set_index('key')
common_keys = pivot_aligned.index.intersection(X_df_indexed.index)

Z = pivot_aligned.loc[common_keys].values
X_aligned = X_df_indexed.loc[common_keys][features].values
Xs_aligned = (X_aligned - scaler_mean) / scaler_scale

# Reconstruct
X_reconstructed = Z @ dictionary

# Compute reconstruction error
mse = np.mean((Xs_aligned - X_reconstructed)**2)
ss_tot = np.sum((Xs_aligned - Xs_aligned.mean(axis=0))**2)
ss_res = np.sum((Xs_aligned - X_reconstructed)**2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
log(f"\nReconstruction MSE (scaled): {mse:.4f}")
log(f"Reconstruction R^2 (scaled): {r2:.4f}")
log(f"(R^2 of 1.0 = perfect reconstruction, 0.0 = as bad as mean, <0 = worse than mean)")

# Per-feature reconstruction
log(f"\n=== PER-FEATURE RECONSTRUCTION R^2 ===")
for i, f in enumerate(features):
    ss_res_f = np.sum((Xs_aligned[:, i] - X_reconstructed[:, i])**2)
    ss_tot_f = np.sum((Xs_aligned[:, i] - Xs_aligned[:, i].mean())**2)
    r2_f = 1 - ss_res_f / ss_tot_f if ss_tot_f > 0 else 0
    log(f"  {f:40s}: R^2 = {r2_f:.3f}")

con.close()

# Write to file
Path('reports/sae_full_analysis.txt').write_text("\n".join(out_lines), encoding='utf-8')
log(f"\nWritten to reports/sae_full_analysis.txt")

#!/usr/bin/env python3
"""Analyze SAE model quality and statistics."""

import duckdb
import json
import numpy as np
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

# Model info
model = con.execute("SELECT * FROM latent_models WHERE model_name = 'sae_apm_v0_k12_a1'").df()
print('=== MODEL INFO ===')
print(f"n_samples: {model.iloc[0]['n_samples']}")
print(f"n_components: {model.iloc[0]['n_components']}")
print(f"alpha: {model.iloc[0]['alpha']}")

features = json.loads(model.iloc[0]['features_json'])
print(f"features ({len(features)}): {features}")

# Dictionary (coefficients) - shape (n_components, n_features)
dictionary = np.array(json.loads(model.iloc[0]['dictionary_json']))
print(f'\n=== DICTIONARY SHAPE ===')
print(f'Shape: {dictionary.shape}')

# Analyze sparsity of dictionary
print(f'\n=== DICTIONARY SPARSITY ===')
for k in range(dictionary.shape[0]):
    row = dictionary[k]
    nonzero = np.sum(np.abs(row) > 0.01)
    max_coef = np.max(np.abs(row))
    print(f'  Dim {k}: {nonzero}/{len(row)} nonzero, max|coef|={max_coef:.3f}')

# Analyze latent skill sparsity
print(f'\n=== LATENT CODE SPARSITY (per season) ===')
ldf = con.execute("""
    SELECT season, dim_idx, 
           COUNT(*) as n,
           AVG(ABS(value)) as mean_abs,
           SUM(CASE WHEN ABS(value) < 0.001 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_zero
    FROM latent_skills
    WHERE model_name = 'sae_apm_v0_k12_a1'
    GROUP BY season, dim_idx
    ORDER BY season, dim_idx
""").df()
print(ldf.to_string())

# Overall stats per dimension
print(f'\n=== OVERALL DIM STATS (pooled) ===')
dim_stats = con.execute("""
    SELECT dim_idx, 
           COUNT(*) as n,
           AVG(value) as mean,
           STDDEV(value) as std,
           AVG(ABS(value)) as mean_abs,
           SUM(CASE WHEN ABS(value) < 0.001 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_zero
    FROM latent_skills
    WHERE model_name = 'sae_apm_v0_k12_a1'
    GROUP BY dim_idx
    ORDER BY dim_idx
""").df()
print(dim_stats.to_string())

# Dimension labels
print(f'\n=== DIMENSION LABELS ===')
labels = con.execute("SELECT dim_idx, label, top_features_json, stable_seasons FROM latent_dim_meta WHERE model_name = 'sae_apm_v0_k12_a1' ORDER BY dim_idx").df()
print(labels.to_string())

# Correlation matrix between dimensions
print(f'\n=== INTER-DIMENSION CORRELATIONS ===')
wide = con.execute("""
    SELECT season, player_id, dim_idx, value
    FROM latent_skills
    WHERE model_name = 'sae_apm_v0_k12_a1'
""").df()
pivot = wide.pivot_table(index=['season', 'player_id'], columns='dim_idx', values='value')
corr = pivot.corr()
print("Correlation matrix (should be near-diagonal for good separation):")
print(corr.round(2).to_string())

con.close()

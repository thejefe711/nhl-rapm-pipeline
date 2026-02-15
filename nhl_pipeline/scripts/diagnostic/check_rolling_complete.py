#!/usr/bin/env python3
"""Check rolling latent skills completion."""

import duckdb
from pathlib import Path

db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
con = duckdb.connect(str(db_path))

print("=" * 60)
print("ROLLING LATENT SKILLS COMPLETION")
print("=" * 60)

df = con.execute("""
    SELECT season,
           COUNT(DISTINCT window_end_game_id) as windows,
           COUNT(*) as total_rows
    FROM rolling_latent_skills
    WHERE model_name = 'sae_apm_v1_k12_a1'
    GROUP BY season
    ORDER BY season
""").df()

if df.empty:
    print("No rolling latent skills found!")
else:
    print(df.to_string(index=False))
    print(f"\nTotal: {df['windows'].sum()} windows, {df['total_rows'].sum()} rows")

con.close()
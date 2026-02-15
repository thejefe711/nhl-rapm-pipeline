#!/usr/bin/env python3
"""Debug percentile calculations."""

import duckdb
import pandas as pd
from pathlib import Path

def debug_scores():
    """Check what the actual scores look like."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get scores for a few key dimensions
    df = con.execute("""
        SELECT
            player_id,
            label,
            value as skill_score
        FROM rolling_latent_skills rls
        LEFT JOIN latent_dim_meta ldm ON rls.model_name = ldm.model_name AND rls.dim_idx = ldm.dim_idx
        WHERE rls.model_name = 'sae_apm_v1_k12_a1'
        AND rls.season = '20242025'
        AND label IN ('Play driver', 'Elite shutdown (HD)', 'Transition killer')
        QUALIFY ROW_NUMBER() OVER (PARTITION BY player_id, rls.dim_idx ORDER BY window_end_game_id DESC) = 1
        ORDER BY label, skill_score DESC
        LIMIT 20
    """).df()

    print("TOP SCORES BY DIMENSION:")
    print("=" * 50)

    for label in df['label'].unique():
        print(f"\n{label}:")
        dim_data = df[df['label'] == label].head(10)
        for _, row in dim_data.iterrows():
            print(".3f")

    # Check McDavid specifically
    mcdavid = con.execute("""
        SELECT
            player_id,
            label,
            value as skill_score
        FROM rolling_latent_skills rls
        LEFT JOIN latent_dim_meta ldm ON rls.model_name = ldm.model_name AND rls.dim_idx = ldm.dim_idx
        WHERE rls.model_name = 'sae_apm_v1_k12_a1'
        AND rls.season = '20242025'
        AND player_id = 8478402
        QUALIFY ROW_NUMBER() OVER (PARTITION BY player_id, rls.dim_idx ORDER BY window_end_game_id DESC) = 1
    """).df()

    print(f"\n\nCONNOR MCDAVID SCORES:")
    print("=" * 50)
    for _, row in mcdavid.iterrows():
        print(".3f")

    con.close()

if __name__ == "__main__":
    debug_scores()
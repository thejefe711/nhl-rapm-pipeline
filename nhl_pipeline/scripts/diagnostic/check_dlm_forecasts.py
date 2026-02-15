#!/usr/bin/env python3
"""Check DLM forecasts completion."""

import duckdb
from pathlib import Path

db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
con = duckdb.connect(str(db_path))

print("=" * 60)
print("DLM FORECASTS COMPLETION")
print("=" * 60)

# Check total forecasts
total_df = con.execute("""
    SELECT
        COUNT(*) as total_forecasts,
        COUNT(DISTINCT player_id) as unique_players,
        COUNT(DISTINCT CONCAT(player_id, '_', dim_idx)) as player_dim_series,
        AVG(n_obs) as avg_obs_per_series,
        MIN(horizon_games) as min_horizon,
        MAX(horizon_games) as max_horizon
    FROM dlm_forecasts
    WHERE model_name = 'sae_apm_v1_k12_a1'
""").df()

if total_df.empty:
    print("No DLM forecasts found!")
else:
    print("Overall summary:")
    print(f"  Total forecasts: {total_df.iloc[0]['total_forecasts']:,}")
    print(f"  Unique players: {total_df.iloc[0]['unique_players']:,}")
    print(f"  Player-dimension series: {total_df.iloc[0]['player_dim_series']:,}")
    print(f"  Avg observations per series: {total_df.iloc[0]['avg_obs_per_series']:.1f}")
    print(f"  Horizons: {total_df.iloc[0]['min_horizon']} to {total_df.iloc[0]['max_horizon']} games")

# Check by season
season_df = con.execute("""
    SELECT
        season,
        COUNT(*) as forecasts,
        COUNT(DISTINCT player_id) as players,
        COUNT(DISTINCT window_end_game_id) as windows
    FROM dlm_forecasts
    WHERE model_name = 'sae_apm_v1_k12_a1'
    GROUP BY season
    ORDER BY season
""").df()

print("\nBy season:")
print(season_df.to_string(index=False))

con.close()
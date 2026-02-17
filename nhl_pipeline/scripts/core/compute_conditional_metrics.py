#!/usr/bin/env python3
"""
Compute advanced conditional metrics:
- Shutdown Score: Performance against elite offensive opponents.
- Breaker Score: Performance against elite defensive opponents.
- Partner Sensitivity Index (PSI): Dependence on teammate quality.
- Elasticity: Rate of improvement with teammate quality.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# CONFIGURATION
# ----------------------------
DUCKDB_PATH = "nhl_pipeline/nhl_canonical.duckdb"
INPUT_TABLE = "shift_context_xg_corsi_positions"
OUTPUT_TABLE = "advanced_player_metrics"

def main():
    con = duckdb.connect(DUCKDB_PATH)
    
    print("Loading shift context data...")
    # Load all shifts
    df = con.execute(f"SELECT * FROM {INPUT_TABLE}").df()
    
    if df.empty:
        print("Error: Input table is empty.")
        return

    print(f"Processing {len(df)} shifts for advanced metrics...")

    # 1. Aggregate per player AND season
    results = []
    
    # Calculate thresholds per season
    season_thresholds = {}
    for season, s_df in df.groupby('season'):
        season_thresholds[season] = {
            'off_elite': s_df['avg_opponent_off_rapm'].quantile(0.8),
            'def_elite': s_df['avg_opponent_def_rapm'].quantile(0.2)
        }
    
    for (player_id, season), p_df in df.groupby(['player_id', 'season']):
        if len(p_df) < 50: # Min 50 shifts for statistical significance
            continue
            
        thresholds = season_thresholds[season]
        
        # Overall Performance
        avg_residual_off = p_df['rapm_residual_xGF'].mean()
        avg_residual_def = p_df['rapm_residual_xGA'].mean()
        
        # Shutdown: performance vs elite offensive opponents
        shutdown_df = p_df[p_df['avg_opponent_off_rapm'] >= thresholds['off_elite']]
        shutdown_score = (shutdown_df['rapm_residual_xGF'] - shutdown_df['rapm_residual_xGA']).mean() if not shutdown_df.empty else 0.0
        
        # Breaker: performance vs elite defensive opponents
        breaker_df = p_df[p_df['avg_opponent_def_rapm'] <= thresholds['def_elite']]
        breaker_score = breaker_df['rapm_residual_xGF'].mean() if not breaker_df.empty else 0.0
        
        # PSI: Correlation between residual and teammate quality
        if len(p_df) > 5 and p_df['avg_teammate_off_rapm'].std() > 0:
            psi = p_df['rapm_residual_xGF'].corr(p_df['avg_teammate_off_rapm'])
            # Elasticity: slope of the regression
            cov = p_df['rapm_residual_xGF'].cov(p_df['avg_teammate_off_rapm'])
            var = p_df['avg_teammate_off_rapm'].var()
            elasticity = cov / var if var > 0 else 0.0
        else:
            psi = 0.0
            elasticity = 0.0
            
        results.append({
            'player_id': player_id,
            'season': season,
            'total_shifts': len(p_df),
            'avg_residual_off': avg_residual_off,
            'avg_residual_def': avg_residual_def,
            'shutdown_score': shutdown_score,
            'breaker_score': breaker_score,
            'psi': psi,
            'elasticity': elasticity
        })
    
    results_df = pd.DataFrame(results)
    
    # Normalize scores (z-score) within each season for better interpretability
    for col in ['shutdown_score', 'breaker_score', 'psi', 'elasticity']:
        results_df[f'{col}_z'] = results_df.groupby('season')[col].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)

    # Store in DuckDB
    print(f"Storing advanced metrics in {OUTPUT_TABLE}...")
    con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
    con.register("adv_temp", results_df)
    con.execute(f"CREATE TABLE {OUTPUT_TABLE} AS SELECT * FROM adv_temp")
    
    # Add names and positions for convenience
    con.execute(f"""
        CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
        SELECT p.full_name, p.position, m.*
        FROM {OUTPUT_TABLE} m
        LEFT JOIN players p ON m.player_id = p.player_id
    """)

    print(f"✅ Advanced metrics table '{OUTPUT_TABLE}' successfully created with {len(results_df)} player-seasons.")
    
    # Get latest season for leaderboards
    latest_season = con.execute(f"SELECT MAX(season) FROM {OUTPUT_TABLE}").fetchone()[0]
    print(f"\nLeaderboards for latest season: {latest_season}")
    
    # Position mapping for leaderboards
    pos_groups = [
        ("Forwards", "('L', 'C', 'R')"),
        ("Defensemen", "('D')")
    ]
    
    for pos_label, pos_in_clause in pos_groups:
        print(f"\n{'='*40}\n{latest_season} {pos_label.upper()} LEADERBOARDS\n{'='*40}")
        
        # Shutdown
        top_shutdown = con.execute(f"""
            SELECT full_name, shutdown_score_z 
            FROM {OUTPUT_TABLE} 
            WHERE season = '{latest_season}' AND position IN {pos_in_clause} 
            ORDER BY shutdown_score_z DESC LIMIT 5
        """).df()
        print(f"\nTop 5 {pos_label} Shutdown Scores:")
        print(top_shutdown.to_string())
        
        # Breaker
        top_breaker = con.execute(f"""
            SELECT full_name, breaker_score_z 
            FROM {OUTPUT_TABLE} 
            WHERE season = '{latest_season}' AND position IN {pos_in_clause} 
            ORDER BY breaker_score_z DESC LIMIT 5
        """).df()
        print(f"\nTop 5 {pos_label} Breaker Scores:")
        print(top_breaker.to_string())

        # Independent (Bottom PSI)
        top_independent = con.execute(f"""
            SELECT full_name, psi_z 
            FROM {OUTPUT_TABLE} 
            WHERE season = '{latest_season}' AND position IN {pos_in_clause} 
            ORDER BY psi_z ASC LIMIT 5
        """).df()
        print(f"\nTop 5 {pos_label} Independent (Low PSI):")
        print(top_independent.to_string())
    
    con.close()

if __name__ == "__main__":
    main()

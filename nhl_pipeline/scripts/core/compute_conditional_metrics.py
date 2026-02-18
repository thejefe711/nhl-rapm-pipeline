#!/usr/bin/env python3
"""
Compute advanced conditional metrics:
- Shutdown Score: Defensive suppression against elite offensive opponents.
- Breaker Score: Offensive output against elite defensive opponents.
- Partner Sensitivity Index (PSI): Dependence on forward linemate quality.
- Elasticity: Rate of change in output per unit of forward linemate quality.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# ----------------------------
# CONFIGURATION
# ----------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent  # scripts/core -> scripts -> nhl_pipeline -> repo root
DUCKDB_PATH = str(_REPO_ROOT / "nhl_pipeline" / "nhl_canonical.duckdb")

INPUT_TABLE = "shift_context_xg_corsi_positions"
OUTPUT_TABLE = "advanced_player_metrics"

# Minimum total TOI (seconds) for a player-season to be included
MIN_TOI_SECONDS = 1800  # 30 minutes

# Elite opponent thresholds (season-level quantiles)
ELITE_OFF_OPPONENT_QUANTILE = 0.80   # top 20% offensive opponents
ELITE_DEF_OPPONENT_QUANTILE = 0.20   # bottom 20% defensive opponents (best defenders)

# PSI split: top/bottom quartile of forward linemate quality
PSI_HIGH_QUANTILE = 0.75
PSI_LOW_QUANTILE  = 0.25


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Duration-weighted mean, returns 0.0 if weights sum to zero."""
    w = weights.values.astype(float)
    v = values.values.astype(float)
    total = w.sum()
    if total <= 0:
        return 0.0
    return float(np.dot(v, w) / total)


def main():
    con = duckdb.connect(DUCKDB_PATH)

    print("Loading shift context data...")
    df = con.execute(f"SELECT * FROM {INPUT_TABLE}").df()

    if df.empty:
        print("Error: Input table is empty.")
        con.close()
        return

    print(f"Processing {len(df):,} shifts for advanced metrics...")

    # Ensure duration column exists and is numeric
    if 'duration_seconds' not in df.columns:
        print("Error: 'duration_seconds' column missing from input table.")
        con.close()
        return
    df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce').fillna(0)

    # -------------------------------------------------------
    # Season-level elite thresholds (shift-weighted quantiles)
    # -------------------------------------------------------
    season_thresholds = {}
    for season, s_df in df.groupby('season'):
        season_thresholds[season] = {
            'off_elite': s_df['avg_opponent_off_rapm'].quantile(ELITE_OFF_OPPONENT_QUANTILE),
            'def_elite': s_df['avg_opponent_def_rapm'].quantile(ELITE_DEF_OPPONENT_QUANTILE),
            'fwd_tm_high': s_df['avg_fwd_teammate_off_rapm'].quantile(PSI_HIGH_QUANTILE),
            'fwd_tm_low':  s_df['avg_fwd_teammate_off_rapm'].quantile(PSI_LOW_QUANTILE),
        }

    # -------------------------------------------------------
    # Per player-season metrics
    # -------------------------------------------------------
    results = []

    for (player_id, season), p_df in df.groupby(['player_id', 'season']):
        # --- Minimum TOI gate (replaces shift-count gate) ---
        total_toi = p_df['duration_seconds'].sum()
        if total_toi < MIN_TOI_SECONDS:
            continue

        thresholds = season_thresholds[season]
        dur = p_df['duration_seconds']

        # --- Overall residuals (duration-weighted) ---
        avg_residual_off = _weighted_mean(p_df['rapm_residual_xGF'], dur)
        avg_residual_def = _weighted_mean(p_df['rapm_residual_xGA'], dur)

        # ---------------------------------------------------
        # SHUTDOWN SCORE
        # Defensive suppression vs elite offensive opponents.
        # Positive = suppressed more xGA than season average.
        # Formula: duration-weighted mean of (-xGA_residual)
        #          on shifts where avg_opponent_off_rapm >= elite threshold.
        # ---------------------------------------------------
        shutdown_df = p_df[p_df['avg_opponent_off_rapm'] >= thresholds['off_elite']]
        n_shutdown_shifts = len(shutdown_df)
        if not shutdown_df.empty:
            shutdown_score = _weighted_mean(
                -shutdown_df['rapm_residual_xGA'],
                shutdown_df['duration_seconds']
            )
            shutdown_consistency = float(shutdown_df['rapm_residual_xGA'].std())
        else:
            shutdown_score = 0.0
            shutdown_consistency = 0.0

        # ---------------------------------------------------
        # BREAKER SCORE
        # Offensive output vs elite defensive opponents.
        # Positive = generated more xGF than season average.
        # Formula: duration-weighted mean of xGF_residual
        #          on shifts where avg_opponent_def_rapm <= elite threshold.
        # ---------------------------------------------------
        breaker_df = p_df[p_df['avg_opponent_def_rapm'] <= thresholds['def_elite']]
        n_breaker_shifts = len(breaker_df)
        if not breaker_df.empty:
            breaker_score = _weighted_mean(
                breaker_df['rapm_residual_xGF'],
                breaker_df['duration_seconds']
            )
        else:
            breaker_score = 0.0

        # ---------------------------------------------------
        # PSI (Partner Sensitivity Index)
        # Measures how much a player's offensive output changes
        # based on forward linemate quality.
        # Formula: mean(xGF_residual | top-quartile fwd linemates)
        #        - mean(xGF_residual | bottom-quartile fwd linemates)
        # Positive = better with good linemates (linemate-dependent).
        # Near zero = produces regardless of linemates (independent).
        # ---------------------------------------------------
        fwd_col = 'avg_fwd_teammate_off_rapm'
        if fwd_col in p_df.columns and p_df[fwd_col].notna().any():
            high_df = p_df[p_df[fwd_col] >= thresholds['fwd_tm_high']]
            low_df  = p_df[p_df[fwd_col] <= thresholds['fwd_tm_low']]
            if not high_df.empty and not low_df.empty:
                psi_upside = _weighted_mean(high_df['rapm_residual_xGF'], high_df['duration_seconds'])
                psi_floor  = _weighted_mean(low_df['rapm_residual_xGF'],  low_df['duration_seconds'])
                psi = psi_upside - psi_floor
            else:
                psi_upside = 0.0
                psi_floor  = 0.0
                psi = 0.0
        else:
            psi_upside = 0.0
            psi_floor  = 0.0
            psi = 0.0

        # ---------------------------------------------------
        # ELASTICITY
        # OLS slope: how much does xGF_residual change per unit
        # of forward linemate RAPM? Includes SE and p-value.
        # Uses scipy.stats.linregress for proper uncertainty.
        # ---------------------------------------------------
        elasticity = 0.0
        elasticity_se = 0.0
        elasticity_pvalue = 1.0

        if fwd_col in p_df.columns:
            valid = p_df[[fwd_col, 'rapm_residual_xGF', 'duration_seconds']].dropna()
            if len(valid) >= 10 and valid[fwd_col].std() > 0:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        valid[fwd_col].values,
                        valid['rapm_residual_xGF'].values
                    )
                    elasticity = float(slope)
                    elasticity_se = float(std_err)
                    elasticity_pvalue = float(p_value)
                except Exception:
                    pass

        # Reliability flag
        is_reliable = total_toi >= MIN_TOI_SECONDS

        results.append({
            'player_id': player_id,
            'season': season,
            'total_shifts': len(p_df),
            'total_toi_seconds': float(total_toi),
            'is_reliable': is_reliable,
            'avg_residual_off': avg_residual_off,
            'avg_residual_def': avg_residual_def,
            # Shutdown
            'shutdown_score': shutdown_score,
            'shutdown_consistency': shutdown_consistency,
            'n_shutdown_shifts': n_shutdown_shifts,
            # Breaker
            'breaker_score': breaker_score,
            'n_breaker_shifts': n_breaker_shifts,
            # PSI
            'psi': psi,
            'psi_upside': psi_upside,
            'psi_floor': psi_floor,
            # Elasticity
            'elasticity': elasticity,
            'elasticity_se': elasticity_se,
            'elasticity_pvalue': elasticity_pvalue,
        })

    if not results:
        print("WARN: No player-seasons met the minimum TOI threshold. Check input data.")
        con.close()
        return

    results_df = pd.DataFrame(results)

    # -------------------------------------------------------
    # Z-score normalization within each season
    # -------------------------------------------------------
    z_cols = ['shutdown_score', 'breaker_score', 'psi', 'elasticity']
    for col in z_cols:
        results_df[f'{col}_z'] = results_df.groupby('season')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )

    # -------------------------------------------------------
    # Store in DuckDB
    # -------------------------------------------------------
    print(f"Storing {len(results_df):,} player-season rows in '{OUTPUT_TABLE}'...")
    con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
    con.register("adv_temp", results_df)
    con.execute(f"CREATE TABLE {OUTPUT_TABLE} AS SELECT * FROM adv_temp")

    # Enrich with player name and position
    con.execute(f"""
        CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
        SELECT p.full_name, p.position, m.*
        FROM {OUTPUT_TABLE} m
        LEFT JOIN players p ON m.player_id = p.player_id
    """)

    print(f"OK Advanced metrics table '{OUTPUT_TABLE}' created with {len(results_df):,} player-seasons.")

    # -------------------------------------------------------
    # Leaderboards for latest season
    # -------------------------------------------------------
    latest_season = con.execute(f"SELECT MAX(season) FROM {OUTPUT_TABLE}").fetchone()[0]
    print(f"\nLeaderboards for latest season: {latest_season}")

    pos_groups = [
        ("Forwards",   "('L', 'C', 'R')"),
        ("Defensemen", "('D')"),
    ]

    for pos_label, pos_in_clause in pos_groups:
        print(f"\n{'='*40}\n{latest_season} {pos_label.upper()} LEADERBOARDS\n{'='*40}")

        for metric, label, order in [
            ('shutdown_score_z', 'Shutdown',    'DESC'),
            ('breaker_score_z',  'Breaker',     'DESC'),
            ('psi_z',            'Independent (Low PSI)', 'ASC'),
            ('elasticity_z',     'Elasticity',  'DESC'),
        ]:
            top = con.execute(f"""
                SELECT full_name, {metric}
                FROM {OUTPUT_TABLE}
                WHERE season = '{latest_season}'
                  AND position IN {pos_in_clause}
                  AND is_reliable = true
                ORDER BY {metric} {order}
                LIMIT 5
            """).df()
            print(f"\nTop 5 {pos_label} {label}:")
            print(top.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()

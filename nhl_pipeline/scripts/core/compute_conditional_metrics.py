#!/usr/bin/env python3
"""
Compute advanced conditional metrics:
- Shutdown Score: Defensive suppression against elite offensive opponents.
- Breaker Score: Offensive output against elite defensive opponents.
- Partner Sensitivity Index (PSI): Dependence on forward linemate quality.
- Two-way PSI: Sensitivity to defensive partner quality (for defenders).
- Clutch Metrics: Performance in close games (score state within 1).
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
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
DUCKDB_PATH = str(_REPO_ROOT / "nhl_pipeline" / "nhl_canonical.duckdb")

INPUT_TABLE = "shift_context_xg_corsi_positions"
OUTPUT_TABLE = "advanced_player_metrics"

# Minimum total TOI (seconds) for a player-season to be included
MIN_TOI_SECONDS = 1800  # 30 minutes

# Elite opponent thresholds (season-level quantiles)
ELITE_OFF_OPPONENT_QUANTILE = 0.80   # top 20% offensive opponents
ELITE_DEF_OPPONENT_QUANTILE = 0.20   # bottom 20% defensive opponents (best defenders)

# PSI split: top/bottom quartile of teammate quality
PSI_HIGH_QUANTILE = 0.75
PSI_LOW_QUANTILE  = 0.25


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Duration-weighted mean, returns 0.0 if weights sum to zero."""
    if values.empty or weights.empty:
        return 0.0
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
    # Season-level thresholds (shift-weighted quantiles)
    # -------------------------------------------------------
    season_thresholds = {}
    for season, s_df in df.groupby('season'):
        season_thresholds[season] = {
            'off_elite': s_df['avg_opponent_off_rapm'].quantile(ELITE_OFF_OPPONENT_QUANTILE),
            'def_elite': s_df['avg_opponent_def_rapm'].quantile(ELITE_DEF_OPPONENT_QUANTILE),
            'fwd_tm_high': s_df['avg_fwd_teammate_off_rapm'].quantile(PSI_HIGH_QUANTILE),
            'fwd_tm_low':  s_df['avg_fwd_teammate_off_rapm'].quantile(PSI_LOW_QUANTILE),
            'def_tm_high': s_df['avg_def_teammate_def_rapm'].quantile(PSI_HIGH_QUANTILE),
            'def_tm_low':  s_df['avg_def_teammate_def_rapm'].quantile(PSI_LOW_QUANTILE),
        }

    # -------------------------------------------------------
    # Per player-season metrics
    # -------------------------------------------------------
    results = []

    for (player_id, season), p_df in df.groupby(['player_id', 'season']):
        total_toi = p_df['duration_seconds'].sum()
        if total_toi < MIN_TOI_SECONDS:
            continue

        thresholds = season_thresholds[season]
        dur = p_df['duration_seconds']

        # --- Overall residuals ---
        avg_residual_off = _weighted_mean(p_df['rapm_residual_xGF'], dur)
        avg_residual_def = _weighted_mean(p_df['rapm_residual_xGA'], dur)

        # --- Shutdown ---
        shutdown_df = p_df[p_df['avg_opponent_off_rapm'] >= thresholds['off_elite']]
        n_shutdown_shifts = len(shutdown_df)
        if not shutdown_df.empty:
            shutdown_score = _weighted_mean(-shutdown_df['rapm_residual_xGA'], shutdown_df['duration_seconds'])
            shutdown_consistency = float(shutdown_df['rapm_residual_xGA'].std())
        else:
            shutdown_score = 0.0
            shutdown_consistency = 0.0

        # --- Breaker ---
        breaker_df = p_df[p_df['avg_opponent_def_rapm'] <= thresholds['def_elite']]
        n_breaker_shifts = len(breaker_df)
        breaker_score = _weighted_mean(breaker_df['rapm_residual_xGF'], breaker_df['duration_seconds']) if not breaker_df.empty else 0.0

        # --- Clutch Metrics (|Score| <= 1) ---
        clutch_df = p_df[p_df['score_state'].abs() <= 1]
        n_clutch_shifts = len(clutch_df)
        if not clutch_df.empty:
            clutch_shutdown = _weighted_mean(-clutch_df['rapm_residual_xGA'], clutch_df['duration_seconds'])
            clutch_breaker = _weighted_mean(clutch_df['rapm_residual_xGF'], clutch_df['duration_seconds'])
        else:
            clutch_shutdown = 0.0
            clutch_breaker = 0.0

        # --- PSI (Partner Sensitivity Index) - Forward Quality ---
        fwd_col = 'avg_fwd_teammate_off_rapm'
        psi_upside, psi_floor, psi = 0.0, 0.0, 0.0
        if fwd_col in p_df.columns:
            high_df = p_df[p_df[fwd_col] >= thresholds['fwd_tm_high']]
            low_df  = p_df[p_df[fwd_col] <= thresholds['fwd_tm_low']]
            if not high_df.empty and not low_df.empty:
                psi_upside = _weighted_mean(high_df['rapm_residual_xGF'], high_df['duration_seconds'])
                psi_floor  = _weighted_mean(low_df['rapm_residual_xGF'], low_df['duration_seconds'])
                psi = psi_upside - psi_floor

        # --- Two-way PSI (Defense Partner Sensitivity) ---
        def_tm_col = 'avg_def_teammate_def_rapm'
        psi_twoway_upside, psi_twoway_floor, psi_twoway = 0.0, 0.0, 0.0
        if def_tm_col in p_df.columns:
            high_def_df = p_df[p_df[def_tm_col] >= thresholds['def_tm_high']]
            low_def_df  = p_df[p_df[def_tm_col] <= thresholds['def_tm_low']]
            if not high_def_df.empty and not low_def_df.empty:
                # Better partner = better suppression (lower xGA). We want positive value for "better with better partner"
                # so: floor (residual with bad partner) - upside (residual with good partner)
                psi_twoway_upside = _weighted_mean(high_def_df['rapm_residual_xGA'], high_def_df['duration_seconds'])
                psi_twoway_floor  = _weighted_mean(low_def_df['rapm_residual_xGA'], low_def_df['duration_seconds'])
                psi_twoway = psi_twoway_floor - psi_twoway_upside

        # --- Elasticity ---
        elasticity, elasticity_se, elasticity_pvalue = 0.0, 0.0, 1.0
        if fwd_col in p_df.columns:
            valid = p_df[[fwd_col, 'rapm_residual_xGF', 'duration_seconds']].dropna()
            if len(valid) >= 10 and valid[fwd_col].std() > 0:
                try:
                    slope, _, _, p_value, std_err = stats.linregress(valid[fwd_col].values, valid['rapm_residual_xGF'].values)
                    elasticity, elasticity_se, elasticity_pvalue = float(slope), float(std_err), float(p_value)
                except Exception: pass

        results.append({
            'player_id': player_id,
            'season': season,
            'total_shifts': len(p_df),
            'total_toi_seconds': float(total_toi),
            'is_reliable': total_toi >= MIN_TOI_SECONDS,
            'avg_residual_off': avg_residual_off,
            'avg_residual_def': avg_residual_def,
            'shutdown_score': shutdown_score,
            'shutdown_consistency': shutdown_consistency,
            'breaker_score': breaker_score,
            'clutch_shutdown': clutch_shutdown,
            'clutch_breaker': clutch_breaker,
            'psi': psi,
            'psi_upside': psi_upside,
            'psi_floor': psi_floor,
            'psi_twoway': psi_twoway,
            'elasticity': elasticity,
            'elasticity_se': elasticity_se,
            'elasticity_pvalue': elasticity_pvalue,
            'n_shutdown_shifts': n_shutdown_shifts,
            'n_breaker_shifts': n_breaker_shifts,
            'n_clutch_shifts': n_clutch_shifts
        })

    if not results:
        print("WARN: No player-seasons met threshold.")
        con.close()
        return

    results_df = pd.DataFrame(results)

    # Z-score normalization
    z_cols = ['shutdown_score', 'breaker_score', 'clutch_shutdown', 'clutch_breaker', 'psi', 'psi_twoway', 'elasticity']
    for col in z_cols:
        results_df[f'{col}_z'] = results_df.groupby('season')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )

    # Store in DuckDB
    print(f"Storing {len(results_df):,} player-season rows in '{OUTPUT_TABLE}'...")
    con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
    con.register("adv_temp", results_df)
    con.execute(f"CREATE TABLE {OUTPUT_TABLE} AS SELECT * FROM adv_temp")

    # Final Enrichment
    con.execute(f"""
        CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
        SELECT p.full_name, p.position, m.*
        FROM {OUTPUT_TABLE} m
        LEFT JOIN players p ON m.player_id = p.player_id
    """)

    con.close()
    print(f"OK Advanced metrics table '{OUTPUT_TABLE}' finished.")

if __name__ == "__main__":
    main()

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

    print(f"Computing thresholds natively in DuckDB...")
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE thresholds AS
        SELECT 
            season,
            QUANTILE_CONT(avg_opponent_off_rapm, {ELITE_OFF_OPPONENT_QUANTILE}) as off_elite,
            QUANTILE_CONT(avg_opponent_def_rapm, {ELITE_DEF_OPPONENT_QUANTILE}) as def_elite,
            QUANTILE_CONT(avg_fwd_teammate_off_rapm, {PSI_HIGH_QUANTILE}) as fwd_tm_high,
            QUANTILE_CONT(avg_fwd_teammate_off_rapm, {PSI_LOW_QUANTILE}) as fwd_tm_low,
            QUANTILE_CONT(avg_def_teammate_def_rapm, {PSI_HIGH_QUANTILE}) as def_tm_high,
            QUANTILE_CONT(avg_def_teammate_def_rapm, {PSI_LOW_QUANTILE}) as def_tm_low
        FROM {INPUT_TABLE}
        GROUP BY season;
    """)

    print(f"Aggregating shifts grouped by player_id and season...")
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE shift_flags AS
        SELECT 
            s.*,
            CASE WHEN s.avg_opponent_off_rapm >= t.off_elite THEN 1 ELSE 0 END as is_shutdown_shift,
            CASE WHEN s.avg_opponent_def_rapm <= t.def_elite THEN 1 ELSE 0 END as is_breaker_shift,
            CASE WHEN ABS(s.score_state) <= 1 THEN 1 ELSE 0 END as is_clutch_shift,
            CASE WHEN s.avg_fwd_teammate_off_rapm >= t.fwd_tm_high THEN 1 ELSE 0 END as is_psi_high_shift,
            CASE WHEN s.avg_fwd_teammate_off_rapm <= t.fwd_tm_low THEN 1 ELSE 0 END as is_psi_low_shift,
            CASE WHEN s.avg_def_teammate_def_rapm >= t.def_tm_high THEN 1 ELSE 0 END as is_psi_twoway_high,
            CASE WHEN s.avg_def_teammate_def_rapm <= t.def_tm_low THEN 1 ELSE 0 END as is_psi_twoway_low
        FROM {INPUT_TABLE} s
        JOIN thresholds t ON s.season = t.season
        WHERE COALESCE(s.duration_seconds, 0) > 0;
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE raw_metrics AS
        SELECT 
            player_id,
            season,
            COUNT(*) as total_shifts,
            SUM(duration_seconds) as total_toi_seconds,
            SUM(duration_seconds) >= {MIN_TOI_SECONDS} as is_reliable,
            
            -- Overall residuals
            SUM(rapm_residual_xGF * duration_seconds) / SUM(duration_seconds) as avg_residual_off,
            SUM(rapm_residual_xGA * duration_seconds) / SUM(duration_seconds) as avg_residual_def,
            
            -- Shutdown
            SUM(is_shutdown_shift) as n_shutdown_shifts,
            COALESCE(SUM(CASE WHEN is_shutdown_shift=1 THEN -rapm_residual_xGA * duration_seconds ELSE 0 END) / 
                     NULLIF(SUM(CASE WHEN is_shutdown_shift=1 THEN duration_seconds ELSE 0 END), 0), 0) as shutdown_score,
            COALESCE(STDDEV_SAMP(CASE WHEN is_shutdown_shift=1 THEN rapm_residual_xGA END), 0) as shutdown_consistency,
            
            -- Breaker
            SUM(is_breaker_shift) as n_breaker_shifts,
            COALESCE(SUM(CASE WHEN is_breaker_shift=1 THEN rapm_residual_xGF * duration_seconds ELSE 0 END) / 
                     NULLIF(SUM(CASE WHEN is_breaker_shift=1 THEN duration_seconds ELSE 0 END), 0), 0) as breaker_score,
                     
            -- Clutch
            SUM(is_clutch_shift) as n_clutch_shifts,
            COALESCE(SUM(CASE WHEN is_clutch_shift=1 THEN -rapm_residual_xGA * duration_seconds ELSE 0 END) / 
                     NULLIF(SUM(CASE WHEN is_clutch_shift=1 THEN duration_seconds ELSE 0 END), 0), 0) as clutch_shutdown,
            COALESCE(SUM(CASE WHEN is_clutch_shift=1 THEN rapm_residual_xGF * duration_seconds ELSE 0 END) / 
                     NULLIF(SUM(CASE WHEN is_clutch_shift=1 THEN duration_seconds ELSE 0 END), 0), 0) as clutch_breaker,
                     
            -- PSI
            COALESCE(SUM(CASE WHEN is_psi_high_shift=1 THEN rapm_residual_xGF * duration_seconds ELSE 0 END) / 
                     NULLIF(SUM(CASE WHEN is_psi_high_shift=1 THEN duration_seconds ELSE 0 END), 0), 0) as psi_upside,
            COALESCE(SUM(CASE WHEN is_psi_low_shift=1 THEN rapm_residual_xGF * duration_seconds ELSE 0 END) / 
                     NULLIF(SUM(CASE WHEN is_psi_low_shift=1 THEN duration_seconds ELSE 0 END), 0), 0) as psi_floor,
            
            -- Two-way PSI
            COALESCE(SUM(CASE WHEN is_psi_twoway_high=1 THEN rapm_residual_xGA * duration_seconds ELSE 0 END) / 
                     NULLIF(SUM(CASE WHEN is_psi_twoway_high=1 THEN duration_seconds ELSE 0 END), 0), 0) as psi_twoway_upside,
            COALESCE(SUM(CASE WHEN is_psi_twoway_low=1 THEN rapm_residual_xGA * duration_seconds ELSE 0 END) / 
                     NULLIF(SUM(CASE WHEN is_psi_twoway_low=1 THEN duration_seconds ELSE 0 END), 0), 0) as psi_twoway_floor,
                     
            -- Elasticity
            CASE WHEN COUNT(rapm_residual_xGF) >= 10 THEN 
                COALESCE(REGR_SLOPE(rapm_residual_xGF, avg_fwd_teammate_off_rapm), 0.0) 
            ELSE 0.0 END as elasticity,
            0.0 as elasticity_se,
            1.0 as elasticity_pvalue
            
        FROM shift_flags
        GROUP BY player_id, season
        HAVING SUM(duration_seconds) >= {MIN_TOI_SECONDS};
    """)

    print(f"Normalizing Z-scores and forming final output table '{OUTPUT_TABLE}'...")
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE scored_metrics AS
        SELECT 
            *,
            (psi_upside - psi_floor) as psi,
            (psi_twoway_floor - psi_twoway_upside) as psi_twoway
        FROM raw_metrics;
        
        DROP TABLE IF EXISTS {OUTPUT_TABLE};
        CREATE TABLE {OUTPUT_TABLE} AS
        SELECT 
            s.*,
            p.full_name,
            p.position,
            COALESCE((shutdown_score - AVG(shutdown_score) OVER (PARTITION BY s.season)) / NULLIF(STDDEV_SAMP(shutdown_score) OVER (PARTITION BY s.season), 0), 0.0) as shutdown_score_z,
            COALESCE((breaker_score - AVG(breaker_score) OVER (PARTITION BY s.season)) / NULLIF(STDDEV_SAMP(breaker_score) OVER (PARTITION BY s.season), 0), 0.0) as breaker_score_z,
            COALESCE((clutch_shutdown - AVG(clutch_shutdown) OVER (PARTITION BY s.season)) / NULLIF(STDDEV_SAMP(clutch_shutdown) OVER (PARTITION BY s.season), 0), 0.0) as clutch_shutdown_z,
            COALESCE((clutch_breaker - AVG(clutch_breaker) OVER (PARTITION BY s.season)) / NULLIF(STDDEV_SAMP(clutch_breaker) OVER (PARTITION BY s.season), 0), 0.0) as clutch_breaker_z,
            COALESCE((psi - AVG(psi) OVER (PARTITION BY s.season)) / NULLIF(STDDEV_SAMP(psi) OVER (PARTITION BY s.season), 0), 0.0) as psi_z,
            COALESCE((psi_twoway - AVG(psi_twoway) OVER (PARTITION BY s.season)) / NULLIF(STDDEV_SAMP(psi_twoway) OVER (PARTITION BY s.season), 0), 0.0) as psi_twoway_z,
            COALESCE((elasticity - AVG(elasticity) OVER (PARTITION BY s.season)) / NULLIF(STDDEV_SAMP(elasticity) OVER (PARTITION BY s.season), 0), 0.0) as elasticity_z
        FROM scored_metrics s
        LEFT JOIN players p ON s.player_id = p.player_id;
    """)

    con.close()
    print(f"OK Advanced metrics table '{OUTPUT_TABLE}' finished.")

if __name__ == "__main__":
    main()

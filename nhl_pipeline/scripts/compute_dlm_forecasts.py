#!/usr/bin/env python3
"""
Compute simple DLM/Kalman forecasts from rolling latent embeddings.

Input:
  DuckDB table: rolling_latent_skills

Output:
  DuckDB table: dlm_forecasts

Model:
  Local-level (random walk) state-space model per (season, player_id, dim_idx):
    x_t = x_{t-1} + w_t,  w_t ~ N(0, q)
    y_t = x_t     + v_t,  v_t ~ N(0, r)

We fit via a straightforward Kalman filter with heuristic q/r:
  r = var(diff(y)) * r_mult
  q = r * q_over_r

This is intentionally minimal and robust for v0. We can upgrade to MLE later.

Usage:
  python compute_dlm_forecasts.py --model sae_apm_v1_k12_a1 --season 20242025 --window 10 --horizons 1,2,3,4,5
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Run: pip install duckdb")
    raise


def _ensure_table(con: "duckdb.DuckDBPyConnection") -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS dlm_forecasts (
            model_name VARCHAR NOT NULL,
            season VARCHAR NOT NULL,
            window_size INTEGER NOT NULL,
            window_end_game_id VARCHAR NOT NULL,
            window_end_time_utc VARCHAR,
            player_id INTEGER NOT NULL,
            dim_idx INTEGER NOT NULL,
            horizon_games INTEGER NOT NULL,
            forecast_mean DOUBLE NOT NULL,
            forecast_var DOUBLE NOT NULL,
            filtered_mean DOUBLE NOT NULL,
            filtered_var DOUBLE NOT NULL,
            n_obs INTEGER NOT NULL,
            q DOUBLE NOT NULL,
            r DOUBLE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (model_name, season, window_size, window_end_game_id, player_id, dim_idx, horizon_games)
        );
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_dlm_player ON dlm_forecasts(player_id, model_name, season);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_dlm_window ON dlm_forecasts(model_name, season, window_size, window_end_game_id);")


def _kalman_filter_last(y: np.ndarray, q: float, r: float) -> Tuple[float, float]:
    """
    Run a local-level Kalman filter and return (filtered_mean, filtered_var) at the last time.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0, 1.0
    if y.size == 1:
        return float(y[0]), float(max(r, 1e-6))

    # init with first observation
    m = float(y[0])
    P = float(max(r, 1e-6))

    for t in range(1, len(y)):
        # predict
        m_pred = m
        P_pred = P + float(max(q, 0.0))
        # update
        S = P_pred + float(max(r, 1e-6))
        K = P_pred / S
        innov = float(y[t] - m_pred)
        m = m_pred + K * innov
        P = (1.0 - K) * P_pred

    return float(m), float(max(P, 1e-12))


def _heuristic_qr(y: np.ndarray, q_over_r: float, r_mult: float) -> Tuple[float, float]:
    """
    Heuristic noise settings based on variability in first differences.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        r = 0.1
        q = r * q_over_r
        return float(q), float(r)

    dy = np.diff(y)
    var_dy = float(np.var(dy))
    # If var is ~0, avoid degeneracy
    r = max(1e-4, var_dy * float(r_mult))
    q = max(1e-6, r * float(q_over_r))
    return float(q), float(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DLM/Kalman forecasts from rolling latents")
    parser.add_argument("--model", type=str, required=True, help="Latent model name (rolling_latent_skills.model_name)")
    parser.add_argument("--season", type=str, default=None, help="Optional season like 20242025 (default: all seasons present)")
    parser.add_argument("--window", type=int, default=10, help="Window size used in rolling_latent_skills (default: 10)")
    parser.add_argument("--horizons", type=str, default="1,2,3,4,5", help="Comma-separated horizons in games (default: 1..5)")
    parser.add_argument("--min-obs", type=int, default=8, help="Min observations required to forecast (default: 8)")
    parser.add_argument("--q-over-r", type=float, default=0.15, help="Process/observation variance ratio (default: 0.15)")
    parser.add_argument("--r-mult", type=float, default=1.0, help="Multiplier on diff variance for r (default: 1.0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite forecasts for the latest window_end_game_id per player")
    args = parser.parse_args()

    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]
    if not horizons:
        raise SystemExit("No horizons provided")

    root = Path(__file__).parent.parent
    db_path = root / "nhl_canonical.duckdb"

    con = duckdb.connect(str(db_path))
    try:
        _ensure_table(con)

        # Determine seasons available
        seasons: List[str]
        if args.season:
            seasons = [str(args.season)]
        else:
            seasons = [r[0] for r in con.execute(
                "SELECT DISTINCT season FROM rolling_latent_skills WHERE model_name = ? AND window_size = ? ORDER BY season",
                [str(args.model), int(args.window)],
            ).fetchall()]

        print("=" * 70)
        print("DLM/KALMAN FORECASTS")
        print("=" * 70)
        print(f"DB: {db_path}")
        print(f"Model: {args.model}")
        print(f"Window size: {int(args.window)}")
        print(f"Horizons: {horizons}")
        print(f"min_obs: {int(args.min_obs)}  q_over_r: {float(args.q_over_r)}  r_mult: {float(args.r_mult)}")

        for season in seasons:
            df = con.execute(
                """
                SELECT season, window_end_game_id, window_end_time_utc, player_id, dim_idx, value
                FROM rolling_latent_skills
                WHERE model_name = ? AND season = ? AND window_size = ?
                """,
                [str(args.model), str(season), int(args.window)],
            ).df()
            if df.empty:
                continue

            df["player_id"] = df["player_id"].astype(int)
            df["dim_idx"] = df["dim_idx"].astype(int)
            df["value"] = df["value"].astype(float)

            # Prefer sorting by window_end_time_utc if available, fallback to window_end_game_id
            if "window_end_time_utc" in df.columns and df["window_end_time_utc"].notna().any():
                df["sort_key"] = df["window_end_time_utc"].fillna(df["window_end_game_id"]).astype(str)
            else:
                df["sort_key"] = df["window_end_game_id"].astype(str)

            print(f"\n--- Season {season}: rows={len(df):,} ---")

            out_rows = []
            grouped = df.sort_values("sort_key").groupby(["player_id", "dim_idx"], sort=False)
            for (pid, k), g in grouped:
                y = g["value"].values.astype(float)
                if len(y) < int(args.min_obs):
                    continue

                q, r = _heuristic_qr(y, q_over_r=float(args.q_over_r), r_mult=float(args.r_mult))
                m_last, P_last = _kalman_filter_last(y, q=q, r=r)

                # Forecasts from last filtered state
                # For random walk: mean stays at m_last; variance increases by h*q
                window_end_game_id = str(g.iloc[-1]["window_end_game_id"])
                window_end_time_utc = g.iloc[-1].get("window_end_time_utc")
                window_end_time_utc = str(window_end_time_utc) if pd.notna(window_end_time_utc) else None

                for h in horizons:
                    out_rows.append(
                        {
                            "model_name": str(args.model),
                            "season": str(season),
                            "window_size": int(args.window),
                            "window_end_game_id": window_end_game_id,
                            "window_end_time_utc": window_end_time_utc,
                            "player_id": int(pid),
                            "dim_idx": int(k),
                            "horizon_games": int(h),
                            "forecast_mean": float(m_last),
                            "forecast_var": float(P_last + float(h) * q),
                            "filtered_mean": float(m_last),
                            "filtered_var": float(P_last),
                            "n_obs": int(len(y)),
                            "q": float(q),
                            "r": float(r),
                        }
                    )

            if not out_rows:
                print("  WARN: No series met min_obs; skipping.")
                continue

            out_df = pd.DataFrame(out_rows)
            if args.overwrite:
                # Overwrite only the specific window_end_game_id values present in out_df (latest per player/dim).
                ends = sorted(set(out_df["window_end_game_id"].astype(str).tolist()))
                placeholders = ",".join(["?"] * len(ends))
                con.execute(
                    f"""
                    DELETE FROM dlm_forecasts
                    WHERE model_name = ? AND season = ? AND window_size = ? AND window_end_game_id IN ({placeholders})
                    """,
                    [str(args.model), str(season), int(args.window), *ends],
                )

            con.execute(
                """
                INSERT OR REPLACE INTO dlm_forecasts
                    (model_name, season, window_size, window_end_game_id, window_end_time_utc, player_id, dim_idx, horizon_games,
                     forecast_mean, forecast_var, filtered_mean, filtered_var, n_obs, q, r)
                SELECT
                    model_name, season, window_size, window_end_game_id, window_end_time_utc, player_id, dim_idx, horizon_games,
                    forecast_mean, forecast_var, filtered_mean, filtered_var, n_obs, q, r
                FROM out_df
                """
            )

            print(f"  OK wrote forecasts: {len(out_df):,} rows (players~={out_df['player_id'].nunique():,})")

        print(f"\nOK Saved forecasts to DuckDB: {db_path}")
    finally:
        con.close()


if __name__ == "__main__":
    main()


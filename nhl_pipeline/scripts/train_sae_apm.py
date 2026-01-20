#!/usr/bin/env python3
"""
Train a sparse "SAE-like" latent skill model on the APM/RAPM suite stored in DuckDB.

We use scikit-learn DictionaryLearning as a pragmatic Sparse Autoencoder substitute:
- Input: per-(season, player_id) feature vector from `apm_results`
- Output: sparse latent codes (n_components dims) per (season, player_id)

This keeps dependencies light (no torch) while still producing sparse latent dimensions.

Usage:
  python train_sae_apm.py --components 12 --alpha 1.0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Run: pip install duckdb")
    raise

from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    # possession / suppression
    "corsi_off_rapm_5v5",
    "corsi_def_rapm_5v5",
    # chance quality
    "xg_off_rapm_5v5",
    "xg_def_rapm_5v5",
    # high-danger chance quality split (thresholded xG v0; see compute_corsi_apm.py)
    "hd_xg_off_rapm_5v5_ge020",
    "hd_xg_def_rapm_5v5_ge020",
    # discipline (keep only one; drawn is deterministic negation right now)
    "penalties_taken_rapm_5v5",
    # transition / puck management proxy
    "turnover_to_xg_swing_rapm_5v5_w10",
    "takeaway_to_xg_swing_rapm_5v5_w10",
    "giveaway_to_xg_swing_rapm_5v5_w10",
    # special teams roles
    "xg_pp_off_rapm",
    "xg_pk_def_rapm",
    "corsi_pp_off_rapm",
    "corsi_pk_def_rapm",
]


def _suggest_label(top_corr_features: list[str]) -> str:
    """
    Heuristic "sticky label" suggester based on which input features most explain a dimension.

    Keep this in sync with report_sae_latents.py (intentionally duplicated to avoid import path issues).
    """
    feats = [str(f).lower() for f in top_corr_features if f]

    def has(substr: str) -> bool:
        return any(substr in f for f in feats)

    # Special teams first (very distinctive)
    if has("pp_off"):
        return "PP quarterback"
    if has("pk_def"):
        return "PK stopper"

    # Defense archetypes
    if has("hd_xg_def"):
        if has("corsi_def") or has("xg_def"):
            return "Elite shutdown (HD)"
        return "HD suppressor"
    if has("corsi_def") and not has("xg_off") and not has("corsi_off"):
        return "Volume suppressor"

    # Transition / mistakes
    if has("turnover_to_xg_swing") or has("giveaway_to_xg_swing") or has("takeaway_to_xg_swing"):
        return "Transition killer"

    # Offense archetypes
    if has("xg_off") and (has("corsi_off") or has("hd_xg_off")):
        return "Play driver"

    return "Two-way profile"


def _ensure_dim_meta_table(con: "duckdb.DuckDBPyConnection") -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS latent_dim_meta (
            model_name VARCHAR NOT NULL,
            dim_idx INTEGER NOT NULL,
            label VARCHAR NOT NULL,
            top_features_json VARCHAR NOT NULL,
            stable_seasons INTEGER NOT NULL,
            seasons_active_json VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (model_name, dim_idx)
        );
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_latent_dim_meta_model ON latent_dim_meta(model_name);")


def _load_feature_matrix(con: "duckdb.DuckDBPyConnection", features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      X_df: index columns [season, player_id] + feature columns
      long_df: raw apm rows (for debugging)
    """
    placeholders = ", ".join(["?"] * len(features))
    long_df = con.execute(
        f"""
        SELECT season, player_id, metric_name, value
        FROM apm_results
        WHERE metric_name IN ({placeholders})
        """,
        features,
    ).df()

    if long_df.empty:
        raise RuntimeError("No rows found in apm_results for requested features")

    long_df["player_id"] = long_df["player_id"].astype(int)
    # Pivot to wide
    X_df = (
        long_df.pivot_table(index=["season", "player_id"], columns="metric_name", values="value", aggfunc="mean")
        .reset_index()
    )

    # Ensure all requested features exist (even if a metric is missing)
    for f in features:
        if f not in X_df.columns:
            X_df[f] = np.nan

    # We require complete cases for v0 (keeps interpretation clean)
    X_df = X_df.dropna(subset=features).copy()
    return X_df, long_df


def _ensure_tables(con: "duckdb.DuckDBPyConnection") -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS latent_models (
            model_name VARCHAR PRIMARY KEY,
            model_type VARCHAR NOT NULL,
            n_components INTEGER NOT NULL,
            alpha DOUBLE NOT NULL,
            features_json VARCHAR NOT NULL,
            scaler_mean_json VARCHAR NOT NULL,
            scaler_scale_json VARCHAR NOT NULL,
            dictionary_json VARCHAR NOT NULL,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            n_samples INTEGER NOT NULL
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS latent_skills (
            model_name VARCHAR NOT NULL,
            season VARCHAR NOT NULL,
            player_id INTEGER NOT NULL,
            dim_idx INTEGER NOT NULL,
            value DOUBLE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (model_name, season, player_id, dim_idx)
        );
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_latent_skills_player ON latent_skills(player_id, model_name);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_latent_skills_season ON latent_skills(season, model_name);")


def _compute_and_store_dim_meta(
    con: "duckdb.DuckDBPyConnection",
    model_name: str,
    n_components: int,
    features: list[str],
    keys: pd.DataFrame,
    X_df: pd.DataFrame,
    Z: np.ndarray,
    active_nonzero_pct: float = 10.0,
    active_p95_abs: float = 0.25,
) -> None:
    """
    Compute per-dimension metadata:
    - suggested label (from pooled top-correlated features)
    - top 3 features (pooled)
    - stability: number of seasons where the dimension is "active" (activation + magnitude)

    This gives the UI a clean "stable stories" layer without recomputing correlations at request time.
    """
    _ensure_dim_meta_table(con)

    df = keys.copy()
    df["season"] = df["season"].astype(str)
    df["player_id"] = df["player_id"].astype(int)

    # Build wide feature matrix aligned with keys
    for f in features:
        if f not in X_df.columns:
            X_df[f] = np.nan

    feat_wide = X_df[["season", "player_id"] + features].copy()

    # Attach Z (latent codes)
    Z_df = pd.DataFrame(Z, columns=[f"dim_{k}" for k in range(n_components)])
    joined = pd.concat([df.reset_index(drop=True), Z_df.reset_index(drop=True)], axis=1)
    joined = joined.merge(feat_wide, on=["season", "player_id"], how="inner")
    joined = joined.dropna(subset=features).copy()
    if joined.empty:
        return

    # Per-season activity stats per dim
    EPS = 1e-3
    season_active: dict[int, list[str]] = {k: [] for k in range(n_components)}
    seasons = sorted(joined["season"].unique().tolist())
    for season in seasons:
        sub = joined[joined["season"] == season]
        for k in range(n_components):
            v = sub[f"dim_{k}"].astype(float).values
            nonzero_pct = 100.0 * float(np.mean(np.abs(v) > EPS))
            p95_abs = float(np.percentile(np.abs(v), 95)) if len(v) else 0.0
            if (nonzero_pct >= active_nonzero_pct) and (p95_abs >= active_p95_abs):
                season_active[k].append(str(season))

    # Pooled correlations per dim for label + top features
    rows = []
    for k in range(n_components):
        corr_rows = []
        y = joined[f"dim_{k}"]
        for f in features:
            c = y.corr(joined[f])
            if pd.notna(c):
                corr_rows.append((f, float(c), abs(float(c))))
        corr_rows.sort(key=lambda x: x[2], reverse=True)
        top_feats = [r[0] for r in corr_rows[:3]]
        label = _suggest_label(top_feats)
        rows.append(
            {
                "model_name": model_name,
                "dim_idx": int(k),
                "label": str(label),
                "top_features_json": json.dumps(top_feats),
                "stable_seasons": int(len(season_active.get(k, []))),
                "seasons_active_json": json.dumps(season_active.get(k, [])),
            }
        )

    out = pd.DataFrame(rows)
    con.execute("DELETE FROM latent_dim_meta WHERE model_name = ?", [model_name])
    con.execute(
        """
        INSERT OR REPLACE INTO latent_dim_meta
            (model_name, dim_idx, label, top_features_json, stable_seasons, seasons_active_json)
        SELECT model_name, dim_idx, label, top_features_json, stable_seasons, seasons_active_json
        FROM out
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sparse latent skills on APM suite (DictionaryLearning)")
    parser.add_argument("--components", type=int, default=12, help="Latent dimensions (default: 12)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Sparsity strength for DictionaryLearning (default: 1.0)")
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES), help="Comma-separated metric_names from apm_results")
    parser.add_argument("--model-name", type=str, default=None, help="Optional explicit model name")
    args = parser.parse_args()

    components = int(args.components)
    alpha = float(args.alpha)
    features = [f.strip() for f in args.features.split(",") if f.strip()]

    root = Path(__file__).parent.parent
    db_path = root / "nhl_canonical.duckdb"
    model_name = args.model_name or f"sae_apm_v0_k{components}_a{alpha:g}"

    print("=" * 70)
    print("SAE (DictionaryLearning) on APM suite")
    print("=" * 70)
    print(f"DB: {db_path}")
    print(f"Model: {model_name}")
    print(f"Components: {components}")
    print(f"Alpha: {alpha}")
    print(f"Features ({len(features)}): {features}")

    con = duckdb.connect(str(db_path))
    try:
        _ensure_tables(con)

        X_df, _ = _load_feature_matrix(con, features)
        if X_df.empty:
            raise RuntimeError("No complete cases after pivot; check metric coverage")

        # Keep identifiers for writing results back
        keys = X_df[["season", "player_id"]].copy()
        X = X_df[features].astype(float).values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        dl = DictionaryLearning(
            n_components=components,
            alpha=alpha,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            random_state=0,
            max_iter=2000,
        )
        Z = dl.fit_transform(Xs)  # shape: (n_samples, n_components)

        # Write model metadata (overwrite)
        model_row = {
            "model_name": model_name,
            "model_type": "dictionary_learning",
            "n_components": int(components),
            "alpha": float(alpha),
            "features_json": json.dumps(features),
            "scaler_mean_json": json.dumps(scaler.mean_.tolist()),
            "scaler_scale_json": json.dumps(scaler.scale_.tolist()),
            "dictionary_json": json.dumps(dl.components_.tolist()),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": int(len(keys)),
        }
        con.execute("DELETE FROM latent_models WHERE model_name = ?", [model_name])
        con.execute(
            """
            INSERT INTO latent_models
                (model_name, model_type, n_components, alpha, features_json, scaler_mean_json, scaler_scale_json, dictionary_json, trained_at, n_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                model_row["model_name"],
                model_row["model_type"],
                model_row["n_components"],
                model_row["alpha"],
                model_row["features_json"],
                model_row["scaler_mean_json"],
                model_row["scaler_scale_json"],
                model_row["dictionary_json"],
                model_row["trained_at"],
                model_row["n_samples"],
            ],
        )

        # Write latent skills (long form)
        rows = []
        for i in range(len(keys)):
            season = str(keys.iloc[i]["season"])
            pid = int(keys.iloc[i]["player_id"])
            for k in range(components):
                rows.append(
                    {
                        "model_name": model_name,
                        "season": season,
                        "player_id": pid,
                        "dim_idx": int(k),
                        "value": float(Z[i, k]),
                    }
                )

        out_df = pd.DataFrame(rows)
        con.execute("DELETE FROM latent_skills WHERE model_name = ?", [model_name])
        con.execute(
            """
            INSERT OR REPLACE INTO latent_skills (model_name, season, player_id, dim_idx, value)
            SELECT model_name, season, player_id, dim_idx, value FROM out_df
            """
        )

        # Dimension metadata (labels + stability across seasons)
        _compute_and_store_dim_meta(
            con=con,
            model_name=model_name,
            n_components=components,
            features=features,
            keys=keys,
            X_df=X_df,
            Z=Z,
        )

        print(f"OK Trained and stored: samples={len(keys):,} dims={components}")
        print("OK Tables: latent_models, latent_skills, latent_dim_meta")
    finally:
        con.close()


if __name__ == "__main__":
    main()


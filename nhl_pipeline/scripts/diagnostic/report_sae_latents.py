#!/usr/bin/env python3
"""
Report: SAE (DictionaryLearning) latent dimensions.

Shows:
- Top/bottom players per latent dimension (for a chosen season)
- Correlations between latent dims and the input APM features used to train the model

Usage:
  python report_sae_latents.py --model sae_apm_v0_k12_a1 --season 20252026 --top 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Run: pip install duckdb")
    raise

# Reuse name loader (DuckDB players table preferred).
from report_corsi_rapm import load_player_names


def _suggest_label(top_corr_features: list[str]) -> str:
    """
    Heuristic "sticky label" suggester based on which input features most explain a dimension.

    This is intentionally simple and deterministic so the UI can reuse it later.
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

    # Default fallback
    return "Two-way profile"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report SAE latent dimensions")
    parser.add_argument("--model", type=str, required=True, help="Model name from latent_models.model_name")
    parser.add_argument("--season", type=str, default=None, help="Season like 20252026 (required unless --seasons is provided)")
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Optional comma-separated seasons for a stability view (e.g. 20242025,20232024). If set, prints per-dim label/top-features per season.",
    )
    parser.add_argument("--top", type=int, default=15, help="Top/bottom N players per dimension")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    db_path = root / "nhl_canonical.duckdb"

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "latent_models" not in tables or "latent_skills" not in tables:
            print("No latent tables found. Run: python scripts/train_sae_apm.py first.")
            return

        mdf = con.execute(
            "SELECT model_name, n_components, alpha, features_json FROM latent_models WHERE model_name = ?",
            [args.model],
        ).df()
        if mdf.empty:
            print(f"Model not found: {args.model!r}")
            return

        n_components = int(mdf.iloc[0]["n_components"])
        features = json.loads(mdf.iloc[0]["features_json"])
        if not isinstance(features, list) or not features:
            print("Model features_json missing/invalid.")
            return

        if args.seasons:
            seasons = [s.strip() for s in str(args.seasons).split(",") if s.strip()]
            if not seasons:
                print("No seasons provided in --seasons.")
                return

            print("=" * 80)
            print(f"SAE STABILITY REPORT  model={args.model}  seasons={','.join(seasons)}")
            print("=" * 80)
            print(f"n_components={n_components}  features={features}")

            # Preload full name map once
            names = load_player_names(con, root=root)
            name_map = {int(r["player_id"]): r["full_name"] for _, r in names.iterrows()}

            for k in range(n_components):
                print("\n" + "-" * 80)
                print(f"Dim {k}")

                for season in seasons:
                    # Latents -> wide for this season
                    ldf = con.execute(
                        """
                        SELECT player_id, dim_idx, value
                        FROM latent_skills
                        WHERE model_name = ? AND season = ?
                        """,
                        [args.model, season],
                    ).df()
                    if ldf.empty:
                        print(f"  {season}: (no latent rows)")
                        continue

                    ldf["player_id"] = ldf["player_id"].astype(int)
                    Z = ldf.pivot_table(index="player_id", columns="dim_idx", values="value", aggfunc="mean")
                    for kk in range(n_components):
                        if kk not in Z.columns:
                            Z[kk] = 0.0
                    Z = Z[[kk for kk in range(n_components)]].copy()

                    # Features -> wide for same season
                    placeholders = ", ".join(["?"] * len(features))
                    fdf = con.execute(
                        f"""
                        SELECT player_id, metric_name, value
                        FROM apm_results
                        WHERE season = ? AND metric_name IN ({placeholders})
                        """,
                        [season, *features],
                    ).df()
                    if fdf.empty:
                        print(f"  {season}: (no feature rows)")
                        continue

                    fdf["player_id"] = fdf["player_id"].astype(int)
                    X = fdf.pivot_table(index="player_id", columns="metric_name", values="value", aggfunc="mean")
                    for f in features:
                        if f not in X.columns:
                            X[f] = None
                    X = X[features].copy()

                    joined = Z.join(X, how="inner").dropna(axis=0, how="any")
                    if joined.empty:
                        print(f"  {season}: (no complete cases)")
                        continue

                    # Correlations for dim k vs all features
                    corr_rows: list[dict[str, Any]] = []
                    for f in features:
                        c = joined[k].corr(joined[f])
                        if pd.notna(c):
                            corr_rows.append({"feature": f, "corr": float(c), "abs_corr": float(abs(c))})
                    if corr_rows:
                        cdf = pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False).head(3)
                        top_feats = cdf["feature"].tolist() if not cdf.empty else []
                    else:
                        top_feats = []
                    label = _suggest_label(top_feats)

                    # Show exemplar players for this dim in this season (top 3)
                    vals = joined[k].copy()
                    top_p = vals.sort_values(ascending=False).head(3)

                    def fmt(pid: int, v: float) -> str:
                        return f"{name_map.get(int(pid), str(pid))} ({int(pid)}): {v:+.3f}"

                    feats_str = ", ".join(top_feats) if top_feats else "(no correlations)"
                    exemplars = "; ".join(fmt(int(pid), float(v)) for pid, v in top_p.items())
                    print(f"  {season}: {label} | {feats_str} | exemplars: {exemplars}")

            return

        if not args.season:
            print("Missing --season (or provide --seasons for stability view).")
            return

        # Latents -> wide (one row per player_id)
        ldf = con.execute(
            """
            SELECT player_id, dim_idx, value
            FROM latent_skills
            WHERE model_name = ? AND season = ?
            """,
            [args.model, args.season],
        ).df()
        if ldf.empty:
            print(f"No latent rows for model={args.model!r} season={args.season!r}")
            return

        ldf["player_id"] = ldf["player_id"].astype(int)
        Z = ldf.pivot_table(index="player_id", columns="dim_idx", values="value", aggfunc="mean")
        # Ensure all dims exist
        for k in range(n_components):
            if k not in Z.columns:
                Z[k] = 0.0
        Z = Z[[k for k in range(n_components)]].copy()

        # Features -> wide for same season
        placeholders = ", ".join(["?"] * len(features))
        fdf = con.execute(
            f"""
            SELECT player_id, metric_name, value
            FROM apm_results
            WHERE season = ? AND metric_name IN ({placeholders})
            """,
            [args.season, *features],
        ).df()
        fdf["player_id"] = fdf["player_id"].astype(int)
        X = fdf.pivot_table(index="player_id", columns="metric_name", values="value", aggfunc="mean")
        for f in features:
            if f not in X.columns:
                X[f] = None
        X = X[features].copy()

        # Join, complete cases for correlations
        joined = Z.join(X, how="inner")
        joined = joined.dropna(axis=0, how="any")
        if joined.empty:
            print("No complete cases after joining latents + features (unexpected).")
            return

        names = load_player_names(con, root=root)
        name_map = {int(r["player_id"]): r["full_name"] for _, r in names.iterrows()}

        # Correlation: each dim vs each feature
        corr_rows: List[Dict[str, Any]] = []
        for k in range(n_components):
            for f in features:
                c = joined[k].corr(joined[f])
                corr_rows.append({"dim": int(k), "feature": f, "corr": float(c) if pd.notna(c) else None})
        corr_df = pd.DataFrame(corr_rows)

        print("=" * 80)
        print(f"SAE LATENT REPORT  model={args.model}  season={args.season}  samples={len(joined)}")
        print("=" * 80)
        print(f"n_components={n_components}  features={features}")

        # For each dim, show top correlations and top/bottom players
        topn = int(args.top)
        for k in range(n_components):
            print("\n" + "-" * 80)
            print(f"Dim {k}")

            sub = corr_df[corr_df["dim"] == k].dropna(subset=["corr"]).copy()
            sub["abs_corr"] = sub["corr"].abs()
            sub = sub.sort_values("abs_corr", ascending=False).head(6)

            label = _suggest_label(sub["feature"].tolist())
            print(f"Suggested label: {label}")
            print("Top feature correlations:")
            if sub.empty:
                print("  (none)")
            else:
                for _, r in sub.iterrows():
                    print(f"  {r['feature']}: {r['corr']:+.3f}")

            # Top/bottom players
            vals = joined[k].copy()
            top_p = vals.sort_values(ascending=False).head(topn)
            bot_p = vals.sort_values(ascending=True).head(topn)

            def fmt(pid: int, v: float) -> str:
                return f"{name_map.get(int(pid), str(pid))} ({int(pid)}): {v:+.3f}"

            print(f"Top {topn} players:")
            for pid, v in top_p.items():
                print("  " + fmt(int(pid), float(v)))

            print(f"Bottom {topn} players:")
            for pid, v in bot_p.items():
                print("  " + fmt(int(pid), float(v)))
    finally:
        con.close()


if __name__ == "__main__":
    main()


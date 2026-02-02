#!/usr/bin/env python3
"""
Pre-compute xG values for all shots in a season.

This script runs the xG model once and saves predictions to Parquet,
enabling fast RAPM reruns without re-predicting xG values.

Usage:
  python scripts/precompute_xg.py --season 20242025
  python scripts/precompute_xg.py --season 20242025 --force-retrain

Output:
  staging/{season}/shots_with_xg.parquet
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

try:
    import joblib
except ImportError:
    print("joblib not installed. Run: pip install joblib")
    raise

from sklearn.linear_model import LogisticRegression


def _xg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build xG features from coordinates."""
    out = pd.DataFrame(index=df.index)
    x = pd.to_numeric(df.get("x_coord"), errors="coerce").astype(float)
    y = pd.to_numeric(df.get("y_coord"), errors="coerce").astype(float)

    ax = x.abs()
    ay = y.abs()
    dx = (89.0 - ax).clip(lower=0.0)
    dist = np.sqrt(dx * dx + ay * ay)
    angle = np.arctan2(ay, dx.replace(0.0, np.nan)).fillna(np.pi / 2.0)

    out["dist"] = dist
    out["angle"] = angle
    out["shot_type"] = df.get("shot_type").fillna("UNKNOWN").astype(str)
    return out


def _train_xg_model(events: pd.DataFrame) -> LogisticRegression:
    """Train xG model on shot events."""
    XG_EVENT_TYPES = {"SHOT", "MISSED_SHOT", "GOAL"}
    
    df = events.copy()
    df = df[df["event_type"].isin(XG_EVENT_TYPES)].copy()
    df = df[df.get("empty_net").fillna(False) == False].copy()
    df = df[pd.notna(df.get("x_coord")) & pd.notna(df.get("y_coord"))].copy()

    print(f"  Training xG model on {len(df):,} shots...")
    if len(df) == 0:
        raise ValueError("No training data for xG model")

    y = (df["event_type"] == "GOAL").astype(int).values
    print(f"  Goals: {sum(y):,} ({100*sum(y)/len(y):.1f}%)")
    
    Xf = _xg_features(df)
    shot_dummies = pd.get_dummies(Xf["shot_type"], prefix="shot", dummy_na=False)
    X = pd.concat([Xf[["dist", "angle"]], shot_dummies], axis=1).fillna(0.0)

    model = LogisticRegression(
        solver="lbfgs", max_iter=1000, class_weight="balanced", C=0.5
    )
    model.fit(X.values, y)
    model._xg_columns = list(X.columns)
    return model


def _predict_xg(model: LogisticRegression, events: pd.DataFrame) -> np.ndarray:
    """Predict xG for events."""
    Xf = _xg_features(events)
    shot_dummies = pd.get_dummies(Xf["shot_type"], prefix="shot", dummy_na=False)
    X = pd.concat([Xf[["dist", "angle"]], shot_dummies], axis=1).fillna(0.0)

    cols = getattr(model, "_xg_columns", None)
    if cols:
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[cols]
    
    return model.predict_proba(X.values)[:, 1]


def main():
    parser = argparse.ArgumentParser(description="Pre-compute xG values for shots")
    parser.add_argument("--season", type=str, required=True, help="Season (e.g., 20242025)")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining even if cached model exists")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    staging_dir = root / "staging" / args.season
    canonical_dir = root / "canonical" / args.season
    model_cache_dir = root / "models"
    model_cache_dir.mkdir(exist_ok=True)
    
    output_path = staging_dir / "shots_with_xg.parquet"

    print("=" * 60)
    print(f"Pre-computing xG for season {args.season}")
    print("=" * 60)

    # Collect all 5v5 shots
    all_shots = []
    game_dirs = sorted(canonical_dir.glob("*_event_on_ice.parquet"))
    
    print(f"Found {len(game_dirs)} games in canonical")
    
    for on_ice_path in game_dirs:
        game_id = on_ice_path.stem.replace("_event_on_ice", "")
        events_path = staging_dir / f"{game_id}_events.parquet"
        
        if not events_path.exists():
            continue
            
        events = pd.read_parquet(events_path)
        on_ice = pd.read_parquet(on_ice_path, columns=["event_id", "is_5v5", "home_skater_count", "away_skater_count"])
        
        merged = events.merge(on_ice, on="event_id", how="left")
        shots = merged[
            (merged["is_5v5"] == True) & 
            (merged["home_skater_count"] == 5) & 
            (merged["away_skater_count"] == 5) &
            (merged["event_type"].isin({"SHOT", "MISSED_SHOT", "GOAL"}))
        ].copy()
        
        if not shots.empty:
            shots["game_id"] = game_id
            all_shots.append(shots)
    
    if not all_shots:
        print("ERROR: No shots found!")
        return
    
    all_shots_df = pd.concat(all_shots, ignore_index=True)
    print(f"Total 5v5 shots: {len(all_shots_df):,}")
    
    # Load or train xG model
    model_path = model_cache_dir / f"xg_model_{args.season}.pkl"
    
    if model_path.exists() and not args.force_retrain:
        print(f"  Loading cached model from {model_path}")
        xg_model = joblib.load(model_path)
    else:
        print("  Training new xG model...")
        xg_model = _train_xg_model(all_shots_df)
        joblib.dump(xg_model, model_path)
        print(f"  Saved model to {model_path}")
    
    # Predict xG for all shots
    print("  Predicting xG values...")
    all_shots_df["xg"] = _predict_xg(xg_model, all_shots_df)
    
    # Save
    output_cols = ["event_id", "game_id", "event_type", "x_coord", "y_coord", "shot_type", "xg"]
    output_df = all_shots_df[[c for c in output_cols if c in all_shots_df.columns]]
    output_df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved {len(output_df):,} shots with xG to:")
    print(f"  {output_path}")
    print(f"\nxG Stats:")
    print(f"  Mean: {output_df['xg'].mean():.3f}")
    print(f"  Std:  {output_df['xg'].std():.3f}")
    print(f"  Min:  {output_df['xg'].min():.3f}")
    print(f"  Max:  {output_df['xg'].max():.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Shared xG model utilities for training, loading, and prediction.

This module is intentionally lightweight and dependency-minimal so both
pipeline loader and RAPM scripts can use one consistent implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

from sklearn.linear_model import LogisticRegression


SHOT_EVENT_TYPES = {"SHOT", "MISSED_SHOT", "GOAL"}


def xg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build core geometric xG features from coordinates."""
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


def xg_design_matrix(df: pd.DataFrame, expected_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Build and align xG model design matrix.

    If expected_columns are provided, missing columns are added with 0.0 and
    output is ordered to exactly match expected column order.
    """
    xf = xg_features(df)
    shot_dummies = pd.get_dummies(xf["shot_type"], prefix="shot", dummy_na=False)
    X = pd.concat([xf[["dist", "angle"]], shot_dummies], axis=1).fillna(0.0)
    if expected_columns:
        for c in expected_columns:
            if c not in X.columns:
                X[c] = 0.0
        X = X[list(expected_columns)]
    return X


def train_xg_model(events: pd.DataFrame) -> LogisticRegression:
    """Train a simple logistic xG model on shot events."""
    df = events.copy()
    df = df[df["event_type"].isin(SHOT_EVENT_TYPES)].copy()
    df = df[df.get("empty_net").fillna(False) == False].copy()
    df = df[pd.notna(df.get("x_coord")) & pd.notna(df.get("y_coord"))].copy()
    if df.empty:
        raise ValueError("No xG training rows after filters.")

    y = (df["event_type"] == "GOAL").astype(int).values
    X = xg_design_matrix(df)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        C=0.5,
        n_jobs=None,
    )
    model.fit(X.values, y)
    model._xg_columns = list(X.columns)  # type: ignore[attr-defined]
    return model


def predict_xg(model: LogisticRegression, events: pd.DataFrame) -> np.ndarray:
    """Predict xG probabilities for events rows."""
    cols = getattr(model, "_xg_columns", None)
    X = xg_design_matrix(events, cols)
    return model.predict_proba(X.values)[:, 1]


def load_xg_model(models_dir: Path, season: Optional[str] = None, allow_season_fallback: bool = True):
    """
    Load xG model with explicit deterministic precedence:
      1) global pooled model: xg_model_global.pkl
      2) season model: xg_model_{season}.pkl (if allowed)
    """
    if joblib is None:
        raise RuntimeError("joblib not installed. Run: pip install joblib")

    global_path = models_dir / "xg_model_global.pkl"
    if global_path.exists():
        return joblib.load(global_path), "global", "global"

    if allow_season_fallback and season:
        season_path = models_dir / f"xg_model_{season}.pkl"
        if season_path.exists():
            return joblib.load(season_path), "season", str(season)

    return None, None, None

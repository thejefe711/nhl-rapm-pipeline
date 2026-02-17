#!/usr/bin/env python3
"""
Compute 5v5 Corsi APM / RAPM using ridge regression.

Two modes:
1) Event-level (fast sanity test): one row per 5v5 shot-attempt event
2) Stint-level (RAPM-style): one row per constant-lineup segment, weighted by duration

Outputs are stored in DuckDB `nhl_canonical.duckdb` in table `apm_results`.

Usage:
  python compute_corsi_apm.py --mode stint   # recommended (RAPM-style per-60)
  python compute_corsi_apm.py --mode event   # sanity test
  python compute_corsi_apm.py --season 20242025
"""


import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Run: pip install duckdb")
    raise

from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import LogisticRegression
try:
    import joblib
except ImportError:
    joblib = None

from xg_model_utils import predict_xg as shared_predict_xg
from xg_model_utils import train_xg_model as shared_train_xg_model

try:
    from nhl_pipeline.src.validation.rapm_statistical_validator import RAPMStatisticalValidator
    from nhl_pipeline.src.validation.deep_validator import DeepValidator
except ImportError as e:
    print(f"DEBUG: ImportError: {e}")
    RAPMStatisticalValidator = None
    DeepValidator = None

CORSI_EVENT_TYPES = {"SHOT", "MISSED_SHOT", "BLOCKED_SHOT", "GOAL"}
XG_EVENT_TYPES = {"SHOT", "MISSED_SHOT", "GOAL"}
TURNOVER_EVENT_TYPES = {"TAKEAWAY", "GIVEAWAY"}
DEF_TRIGGER_EVENT_TYPES = {"BLOCKED_SHOT", "FACEOFF"}
DEFAULT_HD_XG_THRESHOLD = 0.20


def _preflight_required_columns(df: pd.DataFrame, required_cols: List[str], metric_name: str) -> bool:
    """
    Fail fast when a metric depends on missing columns.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  FAIL: Skipping metric '{metric_name}' (missing required columns: {', '.join(sorted(missing))})")
        return False
    return True


def _filter_by_strength(events: pd.DataFrame, strength: str) -> pd.DataFrame:
    """
    Filter events by manpower state using canonical on-ice labels when available.

    We only use this for *event filtering* (to avoid mixing strengths inside a stint interval).
    Stint selection itself is driven by shifts at the midpoint.

    Strength options:
      - 5v5: exactly 5 skaters each side (existing behavior)
      - pp:  man-advantage situations only (e.g., 5v4, 5v3, 4v3). Excludes pulled-goalie 6v5 via goalie presence.
      - all: no additional filtering
    """
    s = (strength or "5v5").strip().lower()
    if s == "all":
        return events

    # If we don't have on-ice labels, we can't safely filter.
    required = {"home_skater_count", "away_skater_count"}
    if not required.issubset(set(events.columns)):
        return events

    out = events.copy()

    # For PP filtering we also require both goalies present to avoid treating pulled-goalie 6v5 as "PP".
    has_goalie_cols = ("home_goalie" in out.columns) and ("away_goalie" in out.columns)

    if s == "5v5":
        out = out[(out["home_skater_count"] == 5) & (out["away_skater_count"] == 5)].copy()
        if "is_5v5" in out.columns:
            out = out[out["is_5v5"] == True].copy()
        return out

    if s == "pp":
        # Typical special-teams states are 5v4/4v5, 5v3/3v5, 4v3/3v4.
        # We exclude anything with >5 skaters (extra attacker), and require unequal skater counts.
        hc = pd.to_numeric(out["home_skater_count"], errors="coerce")
        ac = pd.to_numeric(out["away_skater_count"], errors="coerce")
        out = out[(hc >= 3) & (ac >= 3) & (hc <= 5) & (ac <= 5) & (hc != ac)].copy()
        if has_goalie_cols:
            out = out[pd.notna(out["home_goalie"]) & pd.notna(out["away_goalie"])].copy()
        return out

    # Unknown strength -> no filter (fail open for backwards compat)
    return events


def _train_xg_model(events: pd.DataFrame) -> LogisticRegression:
    """
    Train a simple xG model: P(goal | location, shot_type) at 5v5.

    This is intentionally simple/robust for v0; we can upgrade later.
    """
    return shared_train_xg_model(events)


def _predict_xg(model: LogisticRegression, events: pd.DataFrame) -> np.ndarray:
    preds = shared_predict_xg(model, events)
    print(f"DEBUG: _predict_xg range: min={preds.min():.6f}, max={preds.max():.6f}, avg={preds.mean():.6f}")
    return preds


def _load_boxscore_teams(boxscore_path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not boxscore_path.exists():
        return None, None
    with open(boxscore_path) as f:
        box = json.load(f)
    home_id = box.get("homeTeam", {}).get("id")
    away_id = box.get("awayTeam", {}).get("id")
    return (int(home_id) if home_id is not None else None, int(away_id) if away_id is not None else None)


def _skater_cols(prefix: str) -> List[str]:
    return [f"{prefix}_skater_{i}" for i in range(1, 7)]


def _build_sparse_X_net(
    df: pd.DataFrame,
    player_to_col: Dict[int, int],
    home_cols: List[str],
    away_cols: List[str],
) -> csr_matrix:
    """
    Build sparse design matrix:
      +1 for home skaters on ice, -1 for away skaters on ice

    Fully vectorized: uses numpy fancy indexing with a pre-built lookup array.
    """
    n_rows = len(df)
    row_idx_list: List[np.ndarray] = []
    col_idx_list: List[np.ndarray] = []
    vals_list: List[np.ndarray] = []

    if not player_to_col:
        return csr_matrix((n_rows, 0))

    # Build fast lookup array: player_id -> column index (or -1)
    max_pid = max(player_to_col.keys()) + 1
    pid_lookup = np.full(max_pid, -1, dtype=np.int32)
    for pid, col in player_to_col.items():
        pid_lookup[pid] = col

    # Home skaters: +1
    for col_name in home_cols:
        if col_name not in df.columns:
            continue
        col_vals = df[col_name].values
        valid_mask = pd.notna(col_vals)
        if not valid_mask.any():
            continue
        valid_rows = np.where(valid_mask)[0]
        valid_pids = col_vals[valid_mask].astype(np.int64)
        in_range = valid_pids < max_pid
        valid_rows = valid_rows[in_range]
        valid_pids = valid_pids[in_range]
        mapped = pid_lookup[valid_pids]
        good = mapped >= 0
        row_idx_list.append(valid_rows[good])
        col_idx_list.append(mapped[good])
        vals_list.append(np.ones(good.sum()))

    # Away skaters: -1
    for col_name in away_cols:
        if col_name not in df.columns:
            continue
        col_vals = df[col_name].values
        valid_mask = pd.notna(col_vals)
        if not valid_mask.any():
            continue
        valid_rows = np.where(valid_mask)[0]
        valid_pids = col_vals[valid_mask].astype(np.int64)
        in_range = valid_pids < max_pid
        valid_rows = valid_rows[in_range]
        valid_pids = valid_pids[in_range]
        mapped = pid_lookup[valid_pids]
        good = mapped >= 0
        row_idx_list.append(valid_rows[good])
        col_idx_list.append(mapped[good])
        vals_list.append(np.full(good.sum(), -1.0))

    if row_idx_list:
        all_rows = np.concatenate(row_idx_list)
        all_cols = np.concatenate(col_idx_list)
        all_vals = np.concatenate(vals_list)
    else:
        all_rows = np.array([], dtype=int)
        all_cols = np.array([], dtype=int)
        all_vals = np.array([], dtype=float)

    return csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_rows, len(player_to_col)))


def _build_sparse_X_off_def(
    df: pd.DataFrame,
    player_to_col: Dict[int, int],
    off_cols: List[str],
    def_cols: List[str],
) -> csr_matrix:
    """
    Build sparse design matrix with separate offense/defense coefficients per player.

    Fully vectorized: uses numpy fancy indexing and a pre-built lookup array
    to avoid any Python-level row iteration.
    """
    n_rows = len(df)
    row_idx_list: List[np.ndarray] = []
    col_idx_list: List[np.ndarray] = []
    vals_list: List[np.ndarray] = []

    # Precompute per-row scaling vectors
    off_scales = df["off_scale"].fillna(1.0).values if "off_scale" in df.columns else np.ones(n_rows)
    def_scales = df["def_scale"].fillna(1.0).values if "def_scale" in df.columns else np.ones(n_rows)

    # Build a fast lookup array: player_id -> column index (or -1 if not found)
    if player_to_col:
        max_pid = max(player_to_col.keys()) + 1
        pid_lookup = np.full(max_pid, -1, dtype=np.int32)
        for pid, col in player_to_col.items():
            pid_lookup[pid] = col
    else:
        return csr_matrix((n_rows, 0))

    # Process offense columns (fully vectorized per column)
    for col_name in off_cols:
        if col_name not in df.columns:
            continue
        col_vals = df[col_name].values
        valid_mask = pd.notna(col_vals)
        if not valid_mask.any():
            continue
        valid_rows = np.where(valid_mask)[0]
        valid_pids = col_vals[valid_mask].astype(np.int64)
        # Lookup columns, filtering out unknown players (pid >= max_pid or mapped to -1)
        in_range = valid_pids < max_pid
        valid_rows = valid_rows[in_range]
        valid_pids = valid_pids[in_range]
        mapped = pid_lookup[valid_pids]
        good = mapped >= 0
        row_idx_list.append(valid_rows[good])
        col_idx_list.append(2 * mapped[good])
        vals_list.append(off_scales[valid_rows[good]])

    # Process defense columns (fully vectorized per column)
    for col_name in def_cols:
        if col_name not in df.columns:
            continue
        col_vals = df[col_name].values
        valid_mask = pd.notna(col_vals)
        if not valid_mask.any():
            continue
        valid_rows = np.where(valid_mask)[0]
        valid_pids = col_vals[valid_mask].astype(np.int64)
        in_range = valid_pids < max_pid
        valid_rows = valid_rows[in_range]
        valid_pids = valid_pids[in_range]
        mapped = pid_lookup[valid_pids]
        good = mapped >= 0
        row_idx_list.append(valid_rows[good])
        col_idx_list.append(2 * mapped[good] + 1)
        vals_list.append(def_scales[valid_rows[good]])

    if row_idx_list:
        all_rows = np.concatenate(row_idx_list)
        all_cols = np.concatenate(col_idx_list)
        all_vals = np.concatenate(vals_list)
    else:
        all_rows = np.array([], dtype=int)
        all_cols = np.array([], dtype=int)
        all_vals = np.array([], dtype=float)

    return csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_rows, 2 * len(player_to_col)))


def _collect_players_from_onice(df: pd.DataFrame, home_cols: List[str], away_cols: List[str]) -> List[int]:
    players = pd.concat([df[home_cols], df[away_cols]], axis=0).stack().dropna().astype(int).unique().tolist()
    return players


def _ridge_fit(X: csr_matrix, y: np.ndarray, sample_weight: Optional[np.ndarray], alphas: List[float]) -> Tuple[np.ndarray, float]:
    if not alphas:
        alphas = [1000.0, 10000.0]
    
    # Use RidgeCV if multiple alphas (fallback to sklearn)
    if len(alphas) > 1:
        model = RidgeCV(alphas=alphas, fit_intercept=True, scoring=None)
        model.fit(X, y, sample_weight=sample_weight)
        best_alpha = float(model.alpha_)
        print(f"Optimal alpha: {best_alpha}")
        return model.coef_, best_alpha
    
    # Single alpha: solve via normal equations (much faster for sparse X)
    # Solves: (X^T W X + alpha*I) beta = X^T W y with intercept via centering
    import time as _time
    from scipy.linalg import cho_factor, cho_solve

    t0 = _time.time()
    alpha = alphas[0]
    n, p = X.shape
    w = sample_weight if sample_weight is not None else np.ones(n)
    w_sum = w.sum()

    # Weighted means for centering (handles intercept)
    y_mean = np.dot(w, y) / w_sum
    x_mean = np.asarray(X.T @ w).ravel() / w_sum  # p-vector

    # Center y
    y_c = y - y_mean

    # Compute X^T W X efficiently: bake sqrt(w) into X, then Xw^T @ Xw
    # This avoids creating an explicit n x n diagonal matrix
    sqrt_w = np.sqrt(w)
    Xw = X.multiply(sqrt_w[:, np.newaxis])  # scale each row by sqrt(w_i)
    XtWX = (Xw.T @ Xw).toarray()  # p x p dense - single sparse multiply

    # Apply centering correction
    XtWX -= w_sum * np.outer(x_mean, x_mean)

    # Add ridge penalty
    XtWX += alpha * np.eye(p)

    # Right-hand side: X^T @ (w * y_centered)
    XtWy = np.asarray(X.T @ (w * y_c)).ravel()

    # Solve via Cholesky (positive definite due to ridge penalty)
    try:
        c, low = cho_factor(XtWX)
        beta = cho_solve((c, low), XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.solve(XtWX, XtWy)

    elapsed = _time.time() - t0
    print(f"Optimal alpha: {alpha} (normal eq solve: {elapsed:.1f}s)")
    return beta, alpha


def debug_rapm_inputs(X, y, w, stints_df):
    print("=== RAPM INPUT DEBUGGING ===")
    
    # Shapes
    print(f"\n1. SHAPES")
    print(f"   X: {X.shape}")
    print(f"   y: {len(y)}")
    print(f"   w: {len(w)}")
    print(f"   stints: {len(stints_df)}")
    print(f"   Expected X rows: {2 * len(stints_df)}")
    
    # Target variable y
    print(f"\n2. TARGET VARIABLE y")
    print(f"   min:    {y.min():.6f}")
    print(f"   max:    {y.max():.6f}")
    print(f"   mean:   {y.mean():.6f}")
    print(f"   median: {np.median(y):.6f}")
    print(f"   % zero: {(y == 0).mean() * 100:.1f}%")
    
    if y.mean() > 0.01:
        print("   WARNING: y seems too large for per-second rate")
    
    # Weights w
    print(f"\n3. WEIGHTS w (stint duration)")
    print(f"   min:    {w.min():.1f}")
    print(f"   max:    {w.max():.1f}")
    print(f"   mean:   {w.mean():.1f}")
    print(f"   median: {np.median(w):.1f}")
    
    if np.median(w) < 10:
        print("   WARNING: Median weight very low, stint generation likely broken")
    
    # Design matrix X
    print(f"\n4. DESIGN MATRIX X")
    row_sums = np.array(X.sum(axis=1)).flatten()
    print(f"   Row sums: min={row_sums.min()}, max={row_sums.max()}, mean={row_sums.mean():.1f}")
    print(f"   Expected row sum: 10 (5 off + 5 def players)")
    
    col_sums = np.array(X.sum(axis=0)).flatten()
    print(f"   Col sums: min={col_sums.min():.0f}, max={col_sums.max():.0f}")
    print(f"   Players with 0 appearances: {(col_sums == 0).sum()}")
    
    # Sparsity
    if hasattr(X, 'nnz'):
        sparsity = X.nnz / (X.shape[0] * X.shape[1])
    else:
        sparsity = (X != 0).sum() / X.size
    print(f"   Sparsity (% non-zero): {sparsity * 100:.3f}%")
    
    print("\n5. STINT DURATION CHECK")
    # obs_df might not have 'duration' column directly if it was renamed or calculated, 
    # but usually it comes from 'weight' or 'duration_s' in the original data.
    # In this script, obs_df is built from 'obs' list which has 'weight'.
    # But wait, obs_df has 2 rows per stint.
    # So we should check unique weights or just use weights.
    print(f"   Stint duration median (from weights): {np.median(w):.1f}s")
    print(f"   Stints < 5s: {(w < 5).mean() * 100:.1f}%")
    print(f"   Stints < 2s: {(w < 2).mean() * 100:.1f}%")


def _event_level_rows(
    event_on_ice_df: pd.DataFrame,
    events_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
) -> pd.DataFrame:
    df = event_on_ice_df.copy()

    # Back-compat: if canonical doesn't have event_team_id yet, merge it in
    if "event_team_id" not in df.columns or df["event_team_id"].isna().all():
        df = df.merge(events_df[["event_id", "event_team_id"]], on="event_id", how="left")

    df = df[
        (df["is_5v5"] == True) &
        (df["event_type"].isin(CORSI_EVENT_TYPES)) &
        (df["home_skater_count"] == 5) &
        (df["away_skater_count"] == 5)
    ].copy()

    # Require team attribution
    df = df[pd.notna(df["event_team_id"])].copy()
    df["event_team_id"] = df["event_team_id"].astype(int)

    # Encode y as +1 if home generated attempt else -1
    df["y"] = np.where(df["event_team_id"] == int(home_team_id), 1.0, -1.0)

    # Sanity: drop events that claim neither home nor away (shouldn't happen)
    df = df[df["event_team_id"].isin([int(home_team_id), int(away_team_id)])].copy()

    return df


def _blocked_shot_attribution_sanity(events_df: pd.DataFrame, shifts_df: pd.DataFrame) -> Optional[float]:
    """
    Sanity-check: for BLOCKED_SHOT, does event_team_id match shooter team?
    Returns mismatch rate if computable.
    """
    if "event_team_id" not in events_df.columns:
        return None
    blocked = events_df[events_df["event_type"] == "BLOCKED_SHOT"].copy()
    if blocked.empty:
        return 0.0
    if "player_1_id" not in blocked.columns:
        return None

    player_team = shifts_df.groupby("player_id")["team_id"].first().to_dict()
    shooter_team = blocked["player_1_id"].map(lambda pid: player_team.get(int(pid)) if pd.notna(pid) else None)
    mask = shooter_team.notna() & blocked["event_team_id"].notna()
    if mask.sum() == 0:
        return None
    mismatch = (shooter_team[mask].astype(int).values != blocked.loc[mask, "event_team_id"].astype(int).values)
    return float(mismatch.mean())





def _stint_level_rows_from_events(
    event_on_ice_df: pd.DataFrame,
    events_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    xg_model: Optional[LogisticRegression] = None,
    precomputed_xg: Optional[pd.DataFrame] = None,
    turnover_window_s: int = 10,
    hd_xg_threshold: float = DEFAULT_HD_XG_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate stints from event on-ice data.
    
    A stint = consecutive events with the same 10 skaters on ice.
    This replaces the shift-boundary approach which produced tiny stints.
    """
    
    # --- Setup ---
    home_skater_cols = ["home_skater_1", "home_skater_2", "home_skater_3", "home_skater_4", "home_skater_5"]
    away_skater_cols = ["away_skater_1", "away_skater_2", "away_skater_3", "away_skater_4", "away_skater_5"]
    
    # --- Filter to 5v5 ---
    df = event_on_ice_df.copy()
    print(f"DEBUG: Input rows: {len(df)}")
    if "is_5v5" not in df.columns:
        print("DEBUG: is_5v5 column missing!")
    
    df = df[
        (df["is_5v5"] == True) &
        (df["home_skater_count"] == 5) &
        (df["away_skater_count"] == 5)
    ].copy()
    print(f"DEBUG: Rows after 5v5 filter: {len(df)}")
    
    if df.empty:
        return pd.DataFrame()
    
    # --- Merge event details ---
    event_cols = ["event_id", "event_type", "event_team_id", "player_1_id", "player_2_id", "player_3_id", "x_coord", "y_coord", "shot_type", "secondary_type"]
    event_cols = [c for c in event_cols if c in events_df.columns]
    
    # Avoid duplicate columns
    merge_cols = [c for c in event_cols if c not in df.columns or c == "event_id"]
    if merge_cols:
        df = df.merge(events_df[merge_cols], on="event_id", how="left")
    
    # Polyfill shot_type from secondary_type if needed
    if "shot_type" not in df.columns and "secondary_type" in df.columns:
        df["shot_type"] = df["secondary_type"]
    
    # --- Add xG ---
    if precomputed_xg is not None and not precomputed_xg.empty:
        df = df.merge(precomputed_xg[["event_id", "xg"]], on="event_id", how="left")
        df["xg"] = df["xg"].fillna(0.0)
    elif xg_model is not None:
        shot_mask = df["event_type"].isin(XG_EVENT_TYPES)
        df["xg"] = 0.0
        if shot_mask.any():
            has_coords = shot_mask & df["x_coord"].notna() & df["y_coord"].notna()
            if has_coords.any():
                df.loc[has_coords, "xg"] = _predict_xg(xg_model, df.loc[has_coords])
    else:
        df["xg"] = 0.0
    
    # --- Sort by time ---
    df = df.sort_values(["period", "period_seconds"]).reset_index(drop=True)
    
    # --- Build lineup identifier ---
    def get_lineup(row):
        home = frozenset(int(x) for x in row[home_skater_cols] if pd.notna(x))
        away = frozenset(int(x) for x in row[away_skater_cols] if pd.notna(x))
        return (home, away)
    
    df["lineup"] = df.apply(get_lineup, axis=1)
    
    # --- Detect stint boundaries ---
    df["lineup_changed"] = (df["lineup"] != df["lineup"].shift(1)) | (df["period"] != df["period"].shift(1))
    df["stint_id"] = df["lineup_changed"].cumsum()
    
    # --- Pre-calculate team attribution ---
    df["event_team_id"] = pd.to_numeric(df["event_team_id"], errors="coerce")
    df["is_home_event"] = df["event_team_id"] == home_team_id
    df["is_away_event"] = df["event_team_id"] == away_team_id
    
    # --- Event type masks ---
    df["is_corsi"] = df["event_type"].isin(CORSI_EVENT_TYPES)
    df["is_goal"] = df["event_type"] == "GOAL"
    df["is_shot"] = df["event_type"].isin(XG_EVENT_TYPES)
    df["is_penalty"] = df["event_type"] == "PENALTY"
    df["is_takeaway"] = df["event_type"] == "TAKEAWAY"
    df["is_giveaway"] = df["event_type"] == "GIVEAWAY"
    df["is_faceoff"] = df["event_type"] == "FACEOFF"
    df["is_block"] = df["event_type"] == "BLOCKED_SHOT"
    
    # --- Assists ---
    df["has_a1"] = df["is_goal"] & df["player_2_id"].notna()
    df["has_a2"] = df["is_goal"] & df["player_3_id"].notna()
    
    # --- High danger xG ---
    df["is_hd"] = df["xg"] >= hd_xg_threshold
    
    # --- Calculate xG swings for turnovers/blocks/faceoffs ---
    # For each trigger event, sum xG for/against in the next N seconds
    df["turnover_xg_swing"] = 0.0
    df["takeaway_xg_swing"] = 0.0
    df["giveaway_xg_swing"] = 0.0
    df["block_xg_swing"] = 0.0
    df["faceoff_xg_swing"] = 0.0
    
    if turnover_window_s > 0:
        # Vectorized approach for speed
        times = df["period_seconds"].values
        periods = df["period"].values
        teams = df["event_team_id"].values
        xg_vals = df["xg"].values
        is_shot = df["is_shot"].values
        
        for trigger_col, swing_col in [
            ("is_takeaway", "takeaway_xg_swing"),
            ("is_giveaway", "giveaway_xg_swing"),
            ("is_block", "block_xg_swing"),
            ("is_faceoff", "faceoff_xg_swing"),
        ]:
            trigger_mask = df[trigger_col].values
            trigger_indices = np.where(trigger_mask)[0]
            
            for idx in trigger_indices:
                t0 = times[idx]
                p0 = periods[idx]
                team = teams[idx]
                
                if pd.isna(team):
                    continue
                
                # Find shots in window
                window_mask = (
                    (periods == p0) &
                    (times > t0) &
                    (times <= t0 + turnover_window_s) &
                    is_shot
                )
                
                if not window_mask.any():
                    continue
                
                xg_for = xg_vals[window_mask & (teams == team)].sum()
                xg_against = xg_vals[window_mask & (teams != team) & ~np.isnan(teams)].sum()
                
                df.iat[idx, df.columns.get_loc(swing_col)] = xg_for - xg_against
        
        # Combined turnover swing
        df["turnover_xg_swing"] = df["takeaway_xg_swing"] + df["giveaway_xg_swing"]
    
    # --- Aggregate to stint level ---
    stints = []
    
    for stint_id, stint_df in df.groupby("stint_id"):
        if stint_df.empty:
            continue
        
        # Timing
        period = int(stint_df["period"].iloc[0])
        start_s = float(stint_df["period_seconds"].iloc[0])
        end_s = float(stint_df["period_seconds"].iloc[-1])
        
        # Duration: time until next stint starts (or end of period)
        # We'll adjust this after collecting all stints
        
        # Players
        lineup = stint_df["lineup"].iloc[0]
        home_players = list(lineup[0])
        away_players = list(lineup[1])
        
        # Corsi
        corsi_home = int((stint_df["is_corsi"] & stint_df["is_home_event"]).sum())
        corsi_away = int((stint_df["is_corsi"] & stint_df["is_away_event"]).sum())
        
        # Goals
        goals_home = int((stint_df["is_goal"] & stint_df["is_home_event"]).sum())
        goals_away = int((stint_df["is_goal"] & stint_df["is_away_event"]).sum())
        
        # Assists
        a1_home = int((stint_df["has_a1"] & stint_df["is_home_event"]).sum())
        a1_away = int((stint_df["has_a1"] & stint_df["is_away_event"]).sum())
        a2_home = int((stint_df["has_a2"] & stint_df["is_home_event"]).sum())
        a2_away = int((stint_df["has_a2"] & stint_df["is_away_event"]).sum())
        
        # Penalties
        pen_home = int((stint_df["is_penalty"] & stint_df["is_home_event"]).sum())
        pen_away = int((stint_df["is_penalty"] & stint_df["is_away_event"]).sum())
        
        # xG
        xg_home = float(stint_df.loc[stint_df["is_home_event"] & stint_df["is_shot"], "xg"].sum())
        xg_away = float(stint_df.loc[stint_df["is_away_event"] & stint_df["is_shot"], "xg"].sum())
        
        # xG on assists (for xa metrics)
        xg_a1_home = float(stint_df.loc[stint_df["has_a1"] & stint_df["is_home_event"], "xg"].sum())
        xg_a1_away = float(stint_df.loc[stint_df["has_a1"] & stint_df["is_away_event"], "xg"].sum())
        xg_a2_home = float(stint_df.loc[stint_df["has_a2"] & stint_df["is_home_event"], "xg"].sum())
        xg_a2_away = float(stint_df.loc[stint_df["has_a2"] & stint_df["is_away_event"], "xg"].sum())
        
        # High danger xG
        hd_xg_home = float(stint_df.loc[stint_df["is_hd"] & stint_df["is_home_event"] & stint_df["is_shot"], "xg"].sum())
        hd_xg_away = float(stint_df.loc[stint_df["is_hd"] & stint_df["is_away_event"] & stint_df["is_shot"], "xg"].sum())
        
        # Turnover xG swings
        take_swing_home = float(stint_df.loc[stint_df["is_home_event"], "takeaway_xg_swing"].sum())
        take_swing_away = float(stint_df.loc[stint_df["is_away_event"], "takeaway_xg_swing"].sum())
        give_swing_home = float(stint_df.loc[stint_df["is_home_event"], "giveaway_xg_swing"].sum())
        give_swing_away = float(stint_df.loc[stint_df["is_away_event"], "giveaway_xg_swing"].sum())
        turn_swing_home = float(stint_df.loc[stint_df["is_home_event"], "turnover_xg_swing"].sum())
        turn_swing_away = float(stint_df.loc[stint_df["is_away_event"], "turnover_xg_swing"].sum())
        block_swing_home = float(stint_df.loc[stint_df["is_home_event"], "block_xg_swing"].sum())
        block_swing_away = float(stint_df.loc[stint_df["is_away_event"], "block_xg_swing"].sum())
        face_swing_home = float(stint_df.loc[stint_df["is_home_event"], "faceoff_xg_swing"].sum())
        face_swing_away = float(stint_df.loc[stint_df["is_away_event"], "faceoff_xg_swing"].sum())
        
        try:
            stints.append({
                "stint_id": stint_id,
                "period": period,
                "start_s": start_s,
                "end_s": end_s,
                "home_players": home_players,
                "away_players": away_players,
                
                "corsi_home": corsi_home,
                "corsi_away": corsi_away,
                "goals_home": goals_home,
                "goals_away": goals_away,
                "a1_home": a1_home,
                "a1_away": a1_away,
                "a2_home": a2_home,
                "a2_away": a2_away,
                "pen_taken_home": pen_home,
                "pen_taken_away": pen_away,
                "xg_home": xg_home,
                "xg_away": xg_away,
                "xg_a1_home": xg_a1_home,
                "xg_a1_away": xg_a1_away,
                "xg_a2_home": xg_a2_home,
                "xg_a2_away": xg_a2_away,
                "hd_xg_home": hd_xg_home,
                "hd_xg_away": hd_xg_away,
                "take_xg_swing_home": take_swing_home,
                "take_xg_swing_away": take_swing_away,
                "give_xg_swing_home": give_swing_home,
                "give_xg_swing_away": give_swing_away,
                "turnover_xg_swing_home": turn_swing_home,
                "turnover_xg_swing_away": turn_swing_away,
                "block_xg_swing_home": block_swing_home,
                "block_xg_swing_away": block_swing_away,
                "face_xg_swing_home": face_swing_home,
                "face_xg_swing_away": face_swing_away,
            })
        except Exception as e:
            print(f"DEBUG: Error appending stint: {e}")
            raise


    if not stints:
        return pd.DataFrame()
    
    stints_df = pd.DataFrame(stints)
    
    # --- Calculate duration ---
    stints_df = stints_df.sort_values(["period", "start_s"]).reset_index(drop=True)
    stints_df["next_start"] = stints_df.groupby("period")["start_s"].shift(-1)
    stints_df["duration_s"] = stints_df["next_start"] - stints_df["start_s"]
    
    # Last stint in each period
    last_in_period = stints_df["next_start"].isna()
    stints_df.loc[last_in_period, "duration_s"] = 1200 - stints_df.loc[last_in_period, "start_s"]
    
    # Drop stints with no duration
    stints_df = stints_df[stints_df["duration_s"] > 0].copy()
    
    # --- Calculate net columns ---
    stints_df["net_corsi"] = stints_df["corsi_home"] - stints_df["corsi_away"]
    stints_df["net_goals"] = stints_df["goals_home"] - stints_df["goals_away"]
    stints_df["net_a1"] = stints_df["a1_home"] - stints_df["a1_away"]
    stints_df["net_a2"] = stints_df["a2_home"] - stints_df["a2_away"]
    stints_df["net_pen_taken"] = stints_df["pen_taken_home"] - stints_df["pen_taken_away"]
    stints_df["net_xg"] = stints_df["xg_home"] - stints_df["xg_away"]
    stints_df["net_xg_a1"] = stints_df["xg_a1_home"] - stints_df["xg_a1_away"]
    stints_df["net_xg_a2"] = stints_df["xg_a2_home"] - stints_df["xg_a2_away"]
    stints_df["net_hd_xg"] = stints_df["hd_xg_home"] - stints_df["hd_xg_away"]
    stints_df["net_finishing"] = stints_df["net_goals"] - stints_df["net_xg"]
    stints_df["net_take_xg_swing"] = stints_df["take_xg_swing_home"] - stints_df["take_xg_swing_away"]
    stints_df["net_give_xg_swing"] = stints_df["give_xg_swing_home"] - stints_df["give_xg_swing_away"]
    stints_df["net_turnover_xg_swing"] = stints_df["turnover_xg_swing_home"] - stints_df["turnover_xg_swing_away"]
    stints_df["net_block_xg_swing"] = stints_df["block_xg_swing_home"] - stints_df["block_xg_swing_away"]
    # Canonical name used downstream by faceoff-loss metric fit.
    stints_df["net_faceoff_loss_xg_swing"] = stints_df["face_xg_swing_home"] - stints_df["face_xg_swing_away"]
    # Backward-compatible alias retained for older exploratory scripts.
    stints_df["net_face_xg_swing"] = stints_df["face_xg_swing_home"] - stints_df["face_xg_swing_away"]
    
    # --- Add player columns for compatibility with existing code ---
    home_cols = _skater_cols("home")
    away_cols = _skater_cols("away")
    
    for i, col in enumerate(home_cols):
        stints_df[col] = stints_df["home_players"].apply(lambda x: x[i] if i < len(x) else None)
    
    for i, col in enumerate(away_cols):
        stints_df[col] = stints_df["away_players"].apply(lambda x: x[i] if i < len(x) else None)
    
    # --- Add weight column ---
    stints_df["weight"] = stints_df["duration_s"]
    
    # --- Clean up ---
    stints_df = stints_df.drop(columns=["stint_id", "next_start"], errors="ignore")
    
    return stints_df




def _write_apm_results(
    conn: "duckdb.DuckDBPyConnection",
    season: str,
    metric_name: str,
    coefs: Dict[int, float],
    toi_seconds: Dict[int, int],
    events_count: int,
    games_count: int,
) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS apm_results (
            season VARCHAR NOT NULL,
            metric_name VARCHAR NOT NULL,
            player_id INTEGER NOT NULL,
            value DOUBLE NOT NULL,
            games_count INTEGER,
            toi_seconds INTEGER,
            events_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (season, metric_name, player_id)
        );
        """
    )

    rows = []
    for pid, val in coefs.items():
        rows.append(
            {
                "season": season,
                "metric_name": metric_name,
                "player_id": int(pid),
                "value": float(val),
                "games_count": int(games_count),
                "toi_seconds": int(toi_seconds.get(pid, 0)),
                "events_count": int(events_count),
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows)
    print(f"DEBUG: _write_apm_results writing {len(df)} rows for {metric_name}")
    conn.execute(
        """
        INSERT OR REPLACE INTO apm_results
            (season, metric_name, player_id, value, games_count, toi_seconds, events_count)
        SELECT
            season, metric_name, player_id, value, games_count, toi_seconds, events_count
        FROM df
        """
    )



def process_game_wrapper(
    game_id: str,
    season: str,
    canonical_dir: Path,
    staging_dir: Path,
    raw_dir: Path,
    args_mode: str,
    args_turnover_window: int,
    args_strength: str,
    args_hd_xg_threshold: float,
    xg_model: Optional[LogisticRegression],
    precomputed_xg_path: Optional[Path],
) -> Tuple[Optional[pd.DataFrame], Dict[int, int], int]:
    """
    Process a single game to generate stint/event rows and TOI counts.
    Returns (df, toi_seconds, total_events_contribution).
    """
    # print(f"DEBUG: Starting game {game_id}")  # Uncomment for deep debugging
    try:
        # Load precomputed xG locally to avoid IPC overhead
        precomputed_xg = None
        if precomputed_xg_path and precomputed_xg_path.exists():
            # Load only necessary columns to save memory/time if not already done
            # But read_parquet is fast.
            # IMPORTANT: Filter by game_id to avoid Cartesian product on event_id!
            # We assume precomputed_xg has "game_id" column.
            # If it's a single file for the season, we must filter.
            # If we can't filter efficiently (read entire file), we do it in memory.
            # Ideally we'd use filters=[('game_id', '==', game_id)] but game_id type matters.
            
            # Read full file (it's small, 85k rows) and filter in memory
            full_xg = pd.read_parquet(precomputed_xg_path)
            
            # Ensure game_id types match. 
            # game_id arg is str (e.g. "2024020001"). 
            # Check column type in df.
            if "game_id" in full_xg.columns:
                # Try matching as string first
                precomputed_xg = full_xg[full_xg["game_id"].astype(str) == str(game_id)].copy()
            else:
                # Fallback: if no game_id column, assume it's already filtered? 
                # No, precompute_xg.py saves all shots.
                # If game_id is missing, we can't safely use it.
                print(f"  WARN: precomputed_xg missing 'game_id' column. Skipping xG load.")
                precomputed_xg = None


        on_ice_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
        shifts_path = staging_dir / season / f"{game_id}_shifts.parquet"
        events_path = staging_dir / season / f"{game_id}_events.parquet"
        boxscore_path = raw_dir / season / game_id / "boxscore.json"

        if not on_ice_path.exists() or not shifts_path.exists() or not events_path.exists():
            return None, {}, 0

        event_on_ice_df = pd.read_parquet(on_ice_path)
        shifts_df = pd.read_parquet(shifts_path)
        events_df = pd.read_parquet(events_path)

        home_team_id = None
        away_team_id = None
        if "home_team_id" in event_on_ice_df.columns and "away_team_id" in event_on_ice_df.columns:
            home_team_id = int(event_on_ice_df["home_team_id"].dropna().iloc[0]) if event_on_ice_df["home_team_id"].notna().any() else None
            away_team_id = int(event_on_ice_df["away_team_id"].dropna().iloc[0]) if event_on_ice_df["away_team_id"].notna().any() else None

        if home_team_id is None or away_team_id is None:
            home_team_id, away_team_id = _load_boxscore_teams(boxscore_path)

        if home_team_id is None or away_team_id is None:
            return None, {}, 0

        # Optional attribution sanity for BLOCKED_SHOT
        mismatch_rate = _blocked_shot_attribution_sanity(events_df, shifts_df)
        if mismatch_rate is not None and mismatch_rate > 0.05:
            print(f"  WARN: {season}/{game_id}: BLOCKED_SHOT shooter-team mismatch rate {mismatch_rate:.1%}")

        toi_seconds: Dict[int, int] = {}
        total_events = 0
        df = pd.DataFrame()

        if args_mode == "event":
            df = _event_level_rows(event_on_ice_df, events_df, home_team_id, away_team_id)
            if df.empty:
                return None, {}, 0
            df["weight"] = 1.0
            total_events = len(df)

            home_cols = _skater_cols("home")
            away_cols = _skater_cols("away")
            for _, r in df.iterrows():
                for pid in pd.concat([r[home_cols], r[away_cols]]).dropna().astype(int).tolist():
                    toi_seconds[pid] = toi_seconds.get(pid, 0) + 1

        else:
            stints = _stint_level_rows_from_events(
                event_on_ice_df=event_on_ice_df,
                events_df=events_df,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                xg_model=xg_model,
                precomputed_xg=precomputed_xg,
                turnover_window_s=args_turnover_window,
                hd_xg_threshold=args_hd_xg_threshold,
            )
            if stints.empty:
                return None, {}, 0
            
            stints["weight"] = stints["duration_s"].astype(float)
            df = stints
            total_events = int(stints["corsi_home"].sum() + stints["corsi_away"].sum())

            home_cols = _skater_cols("home")
            away_cols = _skater_cols("away")
            for _, stint in stints.iterrows():
                dur = int(stint["duration_s"])
                players = pd.concat([stint[home_cols], stint[away_cols]]).dropna().astype(int).tolist()
                for pid in players:
                    toi_seconds[pid] = toi_seconds.get(pid, 0) + dur

        return df, toi_seconds, total_events

    except Exception as e:
        import traceback
        with open("error_log.txt", "a") as f:
            f.write(f"ERROR processing game {game_id}:\n")
            f.write(f"Type: {type(e)}\n")
            f.write(f"Repr: {repr(e)}\n")
            traceback.print_exc(file=f)
        print(f"ERROR processing game {game_id}: {repr(e)}")
        return None, {}, 0


def main():
    parser = argparse.ArgumentParser(description="Compute 5v5 Corsi APM/RAPM")
    parser.add_argument("--mode", choices=["event", "stint"], default="stint", help="event=fast sanity; stint=RAPM-style per-60")
    parser.add_argument("--season", type=str, default=None, help="Limit to a specific season directory (e.g., 20242025)")
    parser.add_argument(
        "--strength",
        choices=["5v5", "pp", "all"],
        default="5v5",
        help="Manpower state for stint mode: 5v5 (default), pp (special teams only; excludes pulled-goalie 6v5), all (no filter)",
    )
    parser.add_argument("--min-toi", type=int, default=600, help="Minimum 5v5 TOI seconds to keep player in output")
    parser.add_argument("--alphas", type=str, default="1e3,1e4,1e5", help="Comma-separated ridge alphas")
    parser.add_argument(
        "--metrics",
        type=str,
        default="corsi",
        help="Comma-separated metrics to compute in stint mode. Options: corsi,corsi_offdef,goals,a1,a2,penalties,xg,xg_offdef,hd_xg,hd_xg_offdef,xa,turnover,defense,suite",
    )
    parser.add_argument("--turnover-window", type=int, default=10, help="Seconds lookahead for turnover-to-xG swing features (default: 10)")
    parser.add_argument("--hd-xg-threshold", type=float, default=DEFAULT_HD_XG_THRESHOLD, help="High-danger xG threshold (default: 0.20)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for game processing (default: 1 = sequential)")
    parser.add_argument("--use-precomputed-xg", action="store_true", help="Load xG from precomputed cache (run precompute_xg.py first)")
    parser.add_argument("--force-retrain-xg", action="store_true", help="Retrain pooled global xG model even if cache exists")
    parser.add_argument("--validate", action="store_true", help="Run statistical validation on RAPM results")
    parser.add_argument("--deep-validate", action="store_true", help="Run deep validation (stints, etc.)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of games for testing")
    args = parser.parse_args()



    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    root = Path(__file__).parent.parent.parent
    staging_dir = root / "staging"
    canonical_dir = root / "canonical"
    raw_dir = root / "raw"
    db_path = root / "nhl_canonical.duckdb"
    data_dir = root / "data"

    # Enforce Gate 2 by default: only compute/store RAPM for games that passed on-ice validation.
    allowed_games: Optional[set[tuple[str, str]]] = None
    validation_path = data_dir / "on_ice_validation.json"
    if validation_path.exists():
        try:
            validations = json.loads(validation_path.read_text())
            allowed_games = set()
            for r in validations:
                if r.get("all_passed"):
                    allowed_games.add((str(r.get("season")), str(r.get("game_id"))))
        except Exception as e:
            print(f"WARN: Failed to parse {validation_path}: {e}")
            allowed_games = None
    else:
        print(f"WARN: {validation_path} not found; run validate_on_ice.py first to enforce Gate 2.")

    # Find games
    games: List[Tuple[str, str]] = []
    for events_file in staging_dir.glob("*/*_events.parquet"):
        season = events_file.parent.name
        if args.season and season != args.season:
            continue
        game_id = events_file.stem.replace("_events", "")
        if (canonical_dir / season / f"{game_id}_event_on_ice.parquet").exists() and (staging_dir / season / f"{game_id}_shifts.parquet").exists():
            if allowed_games is None or (season, game_id) in allowed_games:
                games.append((season, game_id))

    # Sort and limit
    games.sort()
    if args.limit:
        print(f"Limiting to first {args.limit} games for testing.")
        games = games[:args.limit]


    if not games:
        if allowed_games is not None:
            print("No games found that passed on-ice validation (Gate 2).")
        else:
            print("No games found. Run parse/build_on_ice first.")
        return

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    if "suite" in metrics:
        metrics = ["corsi", "corsi_offdef", "goals", "a1", "a2", "penalties", "xg", "xg_offdef", "hd_xg", "hd_xg_offdef", "xa", "turnover"]
    if "defense" in metrics:
        # defensive add-ons (beyond xg_def/corsi_def): post-block recovery + faceoff-loss danger
        metrics = [m for m in metrics if m != "defense"] + ["block", "faceoff_loss"]

    print("=" * 70)
    print(f"RAPM suite - mode={args.mode} strength={args.strength} metrics={','.join(metrics)}")
    print("=" * 70)
    print(f"Games: {len(games)}")
    print(f"Min TOI: {args.min_toi}s")
    print(f"Alphas: {alphas}")

    # Pool across games (per season)
    season_groups: Dict[str, List[str]] = {}
    for season, gid in games:
        season_groups.setdefault(season, []).append(gid)

    needs_xg = args.mode == "stint" and any(
        m in metrics for m in ["xg", "xg_offdef", "hd_xg", "hd_xg_offdef", "xa", "turnover", "block", "faceoff_loss"]
    )
    pooled_xg_model: Optional[LogisticRegression] = None
    if needs_xg and not getattr(args, "use_precomputed_xg", False):
        model_cache_dir = root / "models"
        model_cache_dir.mkdir(exist_ok=True)
        global_model_path = model_cache_dir / "xg_model_global.pkl"

        if global_model_path.exists() and not getattr(args, "force_retrain_xg", False):
            if joblib is None:
                raise RuntimeError("joblib not installed. Run: pip install joblib")
            pooled_xg_model = joblib.load(global_model_path)
            print(f"Using cached pooled xG model: {global_model_path}")
        else:
            print("Training pooled global xG model from all eligible games...")
            train_events = []
            for season, game_id in sorted(games):
                events_path = staging_dir / season / f"{game_id}_events.parquet"
                on_ice_path = canonical_dir / season / f"{game_id}_event_on_ice.parquet"
                if not events_path.exists() or not on_ice_path.exists():
                    continue
                ev = pd.read_parquet(events_path)
                on = pd.read_parquet(
                    on_ice_path, columns=["event_id", "is_5v5", "home_skater_count", "away_skater_count"]
                )
                ev = ev.merge(on, on="event_id", how="left")
                ev = ev[(ev["is_5v5"] == True) & (ev["home_skater_count"] == 5) & (ev["away_skater_count"] == 5)].copy()
                train_events.append(ev)

            if train_events:
                pooled_train = pd.concat(train_events, ignore_index=True)
                pooled_xg_model = _train_xg_model(pooled_train)
                print(
                    f"  OK Trained pooled xG model: n={len(pooled_train):,} "
                    f"shots={int(pooled_train['event_type'].isin(XG_EVENT_TYPES).sum()):,}"
                )
                if joblib is None:
                    raise RuntimeError("joblib not installed. Run: pip install joblib")
                joblib.dump(pooled_xg_model, global_model_path)
                print(f"  OK Cached pooled model to {global_model_path}")
            else:
                print("  WARN No pooled xG training rows found; xG metrics may be zero/empty.")

    conn = duckdb.connect(str(db_path))

    for season, game_ids in sorted(season_groups.items(), reverse=True):
        print(f"\n--- Season {season} ({len(game_ids)} games) ---")

        all_rows: List[pd.DataFrame] = []
        toi_seconds: Dict[int, int] = {}
        total_events = 0

        xg_model: Optional[LogisticRegression] = pooled_xg_model
        precomputed_xg_path: Optional[Path] = None  # For fast reruns
        
        if needs_xg:
            # Check for precomputed xG first (fastest path)
            precomputed_path = staging_dir / season / "shots_with_xg.parquet"
            if getattr(args, 'use_precomputed_xg', False) and precomputed_path.exists():
                print(f"  OK Found precomputed xG at {precomputed_path}")
                precomputed_xg_path = precomputed_path

        if args.workers > 1:
            print(f"  Processing {len(game_ids)} games with {args.workers} workers...")
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_game_wrapper,
                        game_id,
                        season,
                        canonical_dir,
                        staging_dir,
                        raw_dir,
                        args.mode,
                        int(args.turnover_window),
                        str(args.strength),
                        float(args.hd_xg_threshold),
                        xg_model,
                        precomputed_xg_path
                    ): game_id for game_id in sorted(game_ids)
                }
                
                # Iterate futures directly to enforce timeout check on every task
                for i, future in enumerate(futures):
                    if i % 50 == 0:
                        print(f"  ... processed {i}/{len(game_ids)} games")
                        # Checkpoint: save accumulated rows to disk to prevent data loss on hang
                        if all_rows:
                            chunk_df = pd.concat(all_rows, ignore_index=True)
                            chunk_path = staging_dir / season / f"rapm_partial_{i}.parquet"
                            chunk_df.to_parquet(chunk_path)
                            print(f"  Saved checkpoint: {chunk_path}")
                            # Clear memory (optional, but good practice)
                            # all_rows = []  # If we clear, we need to load them back at the end.
                            # For now, keep in memory but save backup.
                    
                    game_id = futures[future]
                    try:
                        # Timeout to prevent hanging on bad games (60s is generous for one game)
                        df, game_toi, game_events = future.result(timeout=60)
                        if df is not None and not df.empty:
                            all_rows.append(df)
                            total_events += game_events
                            for pid, dur in game_toi.items():
                                toi_seconds[pid] = toi_seconds.get(pid, 0) + dur
                    except concurrent.futures.TimeoutError:
                        print(f"  ERROR: Game {game_id} timed out (stuck worker). Skipping.")
                    except Exception as e:
                        print(f"  ERROR processing game {game_id}: {e}")


        else:
            print(f"  Processing {len(game_ids)} games sequentially...")
            for i, game_id in enumerate(sorted(game_ids)):
                if i % 50 == 0:
                    print(f"  ... processed {i}/{len(game_ids)} games")
                
                df, game_toi, game_events = process_game_wrapper(
                    game_id,
                    season,
                    canonical_dir,
                    staging_dir,
                    raw_dir,
                    args.mode,
                    int(args.turnover_window),
                    str(args.strength),
                    float(args.hd_xg_threshold),
                    xg_model,
                    precomputed_xg_path
                )
                
                if df is not None and not df.empty:
                    all_rows.append(df)
                    total_events += game_events
                    for pid, dur in game_toi.items():
                        toi_seconds[pid] = toi_seconds.get(pid, 0) + dur


        if not all_rows:
            print("  No usable rows; skipping season.")
            continue

        data = pd.concat(all_rows, ignore_index=True)
        
        print(f"DEBUG: args.deep_validate={args.deep_validate}, DeepValidator={DeepValidator}")
        if args.deep_validate and DeepValidator:
            print("DEBUG: Calling validate_stints...")
            DeepValidator.validate_stints(data)
        home_cols = _skater_cols("home")
        away_cols = _skater_cols("away")

        # Filter players by TOI threshold (stint mode uses real seconds)
        keep_players = {pid for pid, toi in toi_seconds.items() if toi >= args.min_toi}
        if not keep_players:
            print("  No players meet min TOI threshold; skipping season.")
            continue

        # Drop rows where any on-ice player is outside the kept set (fully vectorized with np.isin)
        keep_arr = np.array(list(keep_players), dtype=np.float64)
        mask_ok = np.ones(len(data), dtype=bool)
        for col in home_cols + away_cols:
            col_vals = data[col].values.astype(np.float64)
            not_na = pd.notna(col_vals)
            # For non-NaN values: True if in keep_players, NaN positions stay True
            col_ok = np.where(not_na, np.isin(col_vals, keep_arr), True)
            mask_ok &= col_ok
        data = data[mask_ok].reset_index(drop=True)
        if data.empty:
            print("  All rows removed by min TOI filter; skipping season.")
            continue

        players = _collect_players_from_onice(data, home_cols, away_cols)
        players = [p for p in players if p in keep_players]
        players_sorted = sorted(set(players))
        player_to_col = {pid: i for i, pid in enumerate(players_sorted)}

        if args.mode == "event":
            # Preserve existing behavior (Corsi event-level sanity)
            X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
            y = data["y"].astype(float).values
            w = data["weight"].astype(float).values if "weight" in data.columns else None
            coefs, alpha_used = _ridge_fit(X, y, w, alphas)
            coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
            metric_name = "corsi_apm_5v5_event"
            print(f"  Fit: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g} events~={total_events:,}")
            _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(total_events), int(len(game_ids)))
        else:
            # Stint-mode suite
            strength_suffix = str(args.strength).strip().lower()

            def _suffix_for_5v5_only() -> str:
                return "_5v5" if strength_suffix == "5v5" else f"_{strength_suffix}"

            def _hd_tag() -> str:
                thr = float(args.hd_xg_threshold)
                # e.g. 0.20 -> "ge020"
                k = int(round(thr * 100.0))
                return f"ge{str(k).zfill(3)}"

            def _special_teams_offdef_obs(value_home_col: str, value_away_col: str) -> pd.DataFrame:
                """
                Build one observation per stint: advantaged team offense vs disadvantaged team defense.

                This is designed for PP/PK: offense coefs ~ PP impact; defense coefs ~ PK suppression (sign flipped later).
                """
                off_cols = [f"off_skater_{i}" for i in range(1, 7)]
                def_cols = [f"def_skater_{i}" for i in range(1, 7)]
                obs = []
                for _, s in data.iterrows():
                    dur = float(s["duration_s"])
                    if dur <= 0:
                        continue
                    home_players = [int(x) for x in s[home_cols].dropna().astype(int).tolist()]
                    away_players = [int(x) for x in s[away_cols].dropna().astype(int).tolist()]
                    if not home_players or not away_players:
                        continue
                    # Determine advantaged side (more skaters)
                    if len(home_players) > len(away_players):
                        off_players = home_players
                        def_players = away_players
                        val = float(s.get(value_home_col, 0.0))
                    elif len(away_players) > len(home_players):
                        off_players = away_players
                        def_players = home_players
                        val = float(s.get(value_away_col, 0.0))
                    else:
                        # Not special teams (shouldn't happen when strength=pp), skip
                        continue

                    off_scale = 1.0 / max(1, len(off_players))
                    def_scale = 1.0 / max(1, len(def_players))
                    obs.append(
                        {
                            **{off_cols[i]: (off_players + [None] * 6)[i] for i in range(6)},
                            **{def_cols[i]: (def_players + [None] * 6)[i] for i in range(6)},
                            "y": 3600.0 * (val / dur),
                            "weight": dur,
                            "off_scale": off_scale,
                            "def_scale": def_scale,
                        }
                    )
                return pd.DataFrame(obs), off_cols, def_cols

            if "corsi" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping corsi_rapm for non-5v5.")
                elif _preflight_required_columns(data, ["net_corsi", "duration_s", "weight"], "corsi"):
                    y = data["net_corsi"].astype(float) / data["duration_s"].astype(float)
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coefs = coefs * 3600.0  # Convert to per-60 AFTER fitting
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "corsi_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g} events~={int(data['corsi_home'].sum()+data['corsi_away'].sum()):,}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), int(len(game_ids)))

            if "corsi_offdef" in metrics:
                if strength_suffix == "pp":
                    obs_df, off_cols, def_cols = _special_teams_offdef_obs("corsi_home", "corsi_away")
                    if not obs_df.empty:
                        X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                        coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                        off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                        def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}  # higher=better PK suppression
                        print(f"  Fit [corsi_pp/pk]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                        _write_apm_results(conn, season, "corsi_pp_off_rapm", off_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), int(len(game_ids)))
                        _write_apm_results(conn, season, "corsi_pk_def_rapm", def_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), int(len(game_ids)))
                else:
                    # Build two observations per stint: home offense and away offense (vectorized)
                    off_cols = [f"off_skater_{i}" for i in range(1, 7)]
                    def_cols = [f"def_skater_{i}" for i in range(1, 7)]
                    valid = data[data["duration_s"] > 0].copy()
                    dur = valid["duration_s"].values
                    # Home offense obs: off=home, def=away
                    home_obs = pd.DataFrame()
                    for i, (hc, ac) in enumerate(zip(home_cols, away_cols)):
                        home_obs[off_cols[i]] = valid[hc].values
                        home_obs[def_cols[i]] = valid[ac].values
                    home_obs["y"] = valid["corsi_home"].values / dur
                    home_obs["weight"] = dur
                    # Away offense obs: off=away, def=home
                    away_obs = pd.DataFrame()
                    for i, (hc, ac) in enumerate(zip(home_cols, away_cols)):
                        away_obs[off_cols[i]] = valid[ac].values
                        away_obs[def_cols[i]] = valid[hc].values
                    away_obs["y"] = valid["corsi_away"].values / dur
                    away_obs["weight"] = dur
                    obs_df = pd.concat([home_obs, away_obs], ignore_index=True)
                    # Drop rows missing any players (shouldn't happen, but safe)
                    obs_df = obs_df.dropna(subset=[off_cols[0], def_cols[0]])

                    X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                    coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                    coefs = coefs * 3600.0  # Convert to per-60 AFTER fitting
                    off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                    def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}  # higher=better suppression

                    print(f"  Fit [corsi_off/def]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, "corsi_off_rapm_5v5", off_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), int(len(game_ids)))
                    _write_apm_results(conn, season, "corsi_def_rapm_5v5", def_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), int(len(game_ids)))

            if "goals" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping goals_rapm for non-5v5.")
                elif _preflight_required_columns(data, ["net_goals", "duration_s", "weight"], "goals"):
                    y = data["net_goals"].astype(float) / data["duration_s"].astype(float)
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coefs = coefs * 3600.0
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "goals_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(data['goals_home'].sum() + data['goals_away'].sum()), int(len(game_ids)))

            if "a1" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping a1_rapm for non-5v5.")
                elif _preflight_required_columns(data, ["net_a1", "duration_s", "weight"], "a1"):
                    y = data["net_a1"].astype(float) / data["duration_s"].astype(float)
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coefs = coefs * 3600.0
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "primary_assist_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(data['a1_home'].sum() + data['a1_away'].sum()), int(len(game_ids)))

            if "a2" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping a2_rapm for non-5v5.")
                elif _preflight_required_columns(data, ["net_a2", "duration_s", "weight"], "a2"):
                    y = data["net_a2"].astype(float) / data["duration_s"].astype(float)
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coefs = coefs * 3600.0
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "secondary_assist_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(data['a2_home'].sum() + data['a2_away'].sum()), int(len(game_ids)))

            if "penalties" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping penalties_rapm for non-5v5.")
                else:
                    # Off/def penalty regression - two INDEPENDENT metrics
                    # Off coefficient: player impact on their own team taking penalties
                    # Def coefficient (negated): player impact on opponents taking penalties (= drawing)
                    off_cols = [f"off_skater_{i}" for i in range(1, 7)]
                    def_cols = [f"def_skater_{i}" for i in range(1, 7)]
                    dur = data["duration_s"].astype(float).values
                    valid = data[dur > 0].copy()
                    dur = valid["duration_s"].astype(float).values
                    # Home offense obs: off=home, def=away, y=pen_taken_home/dur
                    home_obs = pd.DataFrame()
                    for i, (hc, ac) in enumerate(zip(home_cols, away_cols)):
                        home_obs[off_cols[i]] = valid[hc].values
                        home_obs[def_cols[i]] = valid[ac].values
                    home_obs["y"] = valid["pen_taken_home"].values / dur
                    home_obs["weight"] = dur
                    # Away offense obs: off=away, def=home, y=pen_taken_away/dur
                    away_obs = pd.DataFrame()
                    for i, (hc, ac) in enumerate(zip(home_cols, away_cols)):
                        away_obs[off_cols[i]] = valid[ac].values
                        away_obs[def_cols[i]] = valid[hc].values
                    away_obs["y"] = valid["pen_taken_away"].values / dur
                    away_obs["weight"] = dur
                    obs_df = pd.concat([home_obs, away_obs], ignore_index=True)
                    obs_df = obs_df.dropna(subset=[off_cols[0], def_cols[0]])

                    X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                    coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                    coefs = coefs * 3600.0
                    # Off = penalties committed by player's team (higher = worse discipline)
                    committed_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                    # Def = penalties drawn from opponents (Positive coef = Opponent takes MORE penalties = Good)
                    # Unlike Corsi/xG Defense where we want to suppress (negate positive coef), here we want to cause (keep positive coef).
                    drawn_map = {pid: float(coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}
                    print(f"  Fit [penalties_off/def]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, "penalties_committed_rapm_5v5", committed_map, toi_seconds, int(data['pen_taken_home'].sum() + data['pen_taken_away'].sum()), int(len(game_ids)))
                    _write_apm_results(conn, season, "penalties_drawn_rapm_5v5", drawn_map, toi_seconds, int(data['pen_taken_home'].sum() + data['pen_taken_away'].sum()), int(len(game_ids)))

            if "xg" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping xg_rapm for non-5v5.")
                elif _preflight_required_columns(data, ["net_xg", "duration_s", "weight"], "xg"):
                    # Net xG RAPM
                    y = data["net_xg"].astype(float) / data["duration_s"].astype(float)
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coefs = coefs * 3600.0
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "xg_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "xg_offdef" in metrics:
                if strength_suffix == "pp":
                    obs_df, off_cols, def_cols = _special_teams_offdef_obs("xg_home", "xg_away")
                    if not obs_df.empty:
                        X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                        coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                        off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                        def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}  # higher=better PK suppression
                        print(f"  Fit [xg_pp/pk]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                        _write_apm_results(conn, season, "xg_pp_off_rapm", off_map, toi_seconds, int(len(data)), int(len(game_ids)))
                        _write_apm_results(conn, season, "xg_pk_def_rapm", def_map, toi_seconds, int(len(data)), int(len(game_ids)))
                else:
                    # Two observations per stint: home offense and away offense (vectorized)
                    off_cols = [f"off_skater_{i}" for i in range(1, 7)]
                    def_cols = [f"def_skater_{i}" for i in range(1, 7)]
                    valid = data[data["duration_s"] > 0].copy()
                    dur = valid["duration_s"].values
                    home_obs = pd.DataFrame()
                    for i, (hc, ac) in enumerate(zip(home_cols, away_cols)):
                        home_obs[off_cols[i]] = valid[hc].values
                        home_obs[def_cols[i]] = valid[ac].values
                    home_obs["y"] = valid["xg_home"].values / dur
                    home_obs["weight"] = dur
                    away_obs = pd.DataFrame()
                    for i, (hc, ac) in enumerate(zip(home_cols, away_cols)):
                        away_obs[off_cols[i]] = valid[ac].values
                        away_obs[def_cols[i]] = valid[hc].values
                    away_obs["y"] = valid["xg_away"].values / dur
                    away_obs["weight"] = dur
                    obs_df = pd.concat([home_obs, away_obs], ignore_index=True).dropna(subset=[off_cols[0], def_cols[0]])
                    if not obs_df.empty:
                        X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                        
                        if args.limit:
                             debug_rapm_inputs(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), obs_df)

                        coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                        coefs = coefs * 3600.0
                        off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                        def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}
                        print(f"  Fit [xg_off/def]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                        _write_apm_results(conn, season, "xg_off_rapm_5v5", off_map, toi_seconds, int(len(data)), int(len(game_ids)))
                        _write_apm_results(conn, season, "xg_def_rapm_5v5", def_map, toi_seconds, int(len(data)), int(len(game_ids)))

                        if args.validate and RAPMStatisticalValidator:
                            print("  Running statistical validation for xG RAPM...")
                            class MockModel:
                                def __init__(self, alpha): 
                                    self.alpha = alpha
                                    self.alpha_ = alpha
                                def predict(self, X): 
                                    # This is a placeholder; actual prediction requires unscaled coefs
                                    return X @ (coefs / 3600.0)

                            validator = RAPMStatisticalValidator(
                                X=X, 
                                y=obs_df["y"].values.astype(float), 
                                w=obs_df["weight"].values.astype(float),
                                model=MockModel(alpha_used),
                                coefficients=coefs, 
                                player_mapping={pid: str(pid) for pid in players_sorted}
                            )
                            report = validator.full_validation()
                            print(f"  Validation Report: {report}")
                            with open(f"rapm_validation_report_{season}.txt", "w") as f:
                                f.write(str(report))

            if "hd_xg" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping hd_xg_rapm for non-5v5.")
                elif _preflight_required_columns(data, ["net_hd_xg", "duration_s", "weight"], "hd_xg"):
                    y = data["net_hd_xg"].astype(float) / data["duration_s"].astype(float)
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coefs = coefs * 3600.0
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"hd_xg_rapm_5v5_{_hd_tag()}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "hd_xg_offdef" in metrics:
                tag = _hd_tag()
                if strength_suffix == "pp":
                    obs_df, off_cols, def_cols = _special_teams_offdef_obs("hd_xg_home", "hd_xg_away")
                    if not obs_df.empty:
                        X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                        coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                        off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                        def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}
                        print(f"  Fit [hd_xg_pp/pk_{tag}]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                        _write_apm_results(conn, season, f"hd_xg_pp_off_rapm_{tag}", off_map, toi_seconds, int(len(data)), int(len(game_ids)))
                        _write_apm_results(conn, season, f"hd_xg_pk_def_rapm_{tag}", def_map, toi_seconds, int(len(data)), int(len(game_ids)))
                else:
                    off_cols = [f"off_skater_{i}" for i in range(1, 7)]
                    def_cols = [f"def_skater_{i}" for i in range(1, 7)]
                    valid = data[data["duration_s"] > 0].copy()
                    dur = valid["duration_s"].values
                    home_obs = pd.DataFrame()
                    for i, (hc, ac) in enumerate(zip(home_cols, away_cols)):
                        home_obs[off_cols[i]] = valid[hc].values
                        home_obs[def_cols[i]] = valid[ac].values
                    home_obs["y"] = 3600.0 * (valid["hd_xg_home"].values / dur)
                    home_obs["weight"] = dur
                    away_obs = pd.DataFrame()
                    for i, (hc, ac) in enumerate(zip(home_cols, away_cols)):
                        away_obs[off_cols[i]] = valid[ac].values
                        away_obs[def_cols[i]] = valid[hc].values
                    away_obs["y"] = 3600.0 * (valid["hd_xg_away"].values / dur)
                    away_obs["weight"] = dur
                    obs_df = pd.concat([home_obs, away_obs], ignore_index=True).dropna(subset=[off_cols[0], def_cols[0]])
                    if not obs_df.empty:
                        X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
                        coefs, alpha_used = _ridge_fit(X, obs_df["y"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
                        off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
                        def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}
                        print(f"  Fit [hd_xg_off/def_{tag}]: rows={len(obs_df):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                        _write_apm_results(conn, season, f"hd_xg_off_rapm_5v5_{tag}", off_map, toi_seconds, int(len(data)), int(len(game_ids)))
                        _write_apm_results(conn, season, f"hd_xg_def_rapm_5v5_{tag}", def_map, toi_seconds, int(len(data)), int(len(game_ids)))
            if "xa" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping xa_rapm for non-5v5.")
                elif _preflight_required_columns(data, ["net_xg_a1", "net_xg_a2", "duration_s", "weight"], "xa"):
                    # Proxy xA = xG on assisted GOAL events only (public PBP limitation).
                    # This is intentionally labeled as "on_goals" to avoid over-claiming.
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)

                    y = 3600.0 * (data["net_xg_a1"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "xg_primary_assist_on_goals_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

                    y = 3600.0 * (data["net_xg_a2"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = "xg_secondary_assist_on_goals_rapm_5v5"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "turnover" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping turnover_xg_swing for non-5v5.")
                elif _preflight_required_columns(
                    data,
                    ["net_take_xg_swing", "net_give_xg_swing", "net_turnover_xg_swing", "duration_s", "weight"],
                    "turnover",
                ):
                    # Turnover-triggered xG swing (positive = creates more xG than allows after turnover)
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)

                    y = 3600.0 * (data["net_take_xg_swing"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"takeaway_to_xg_swing_rapm_5v5_w{int(args.turnover_window)}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

                    y = 3600.0 * (data["net_give_xg_swing"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"giveaway_to_xg_swing_rapm_5v5_w{int(args.turnover_window)}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

                    y = 3600.0 * (data["net_turnover_xg_swing"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"turnover_to_xg_swing_rapm_5v5_w{int(args.turnover_window)}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "block" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping block_xg_swing for non-5v5.")
                elif _preflight_required_columns(data, ["net_block_xg_swing", "duration_s", "weight"], "block"):
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    y = 3600.0 * (data["net_block_xg_swing"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"blocked_shot_to_xg_swing_rapm_5v5_w{int(args.turnover_window)}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

            if "faceoff_loss" in metrics:
                if strength_suffix != "5v5":
                    print("  WARN: net RAPM targets are currently only supported for 5v5; skipping faceoff_loss_xg_swing for non-5v5.")
                elif _preflight_required_columns(data, ["net_faceoff_loss_xg_swing", "duration_s", "weight"], "faceoff_loss"):
                    X = _build_sparse_X_net(data, player_to_col, home_cols, away_cols)
                    y = 3600.0 * (data["net_faceoff_loss_xg_swing"].astype(float) / data["duration_s"].astype(float))
                    coefs, alpha_used = _ridge_fit(X, y.values, data["weight"].values, alphas)
                    coef_map = {pid: float(coefs[player_to_col[pid]]) for pid in players_sorted}
                    metric_name = f"faceoff_loss_to_xg_swing_rapm_5v5_w{int(args.turnover_window)}"
                    print(f"  Fit [{metric_name}]: rows={len(data):,} players={len(players_sorted):,} alpha={alpha_used:g}")
                    _write_apm_results(conn, season, metric_name, coef_map, toi_seconds, int(len(data)), int(len(game_ids)))

    conn.close()
    print(f"\nOK Saved to DuckDB: {db_path}")


if __name__ == "__main__":
    main()

